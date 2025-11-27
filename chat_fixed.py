import os, random, json, sqlite3, re, time
os.environ["TRANSFORMERS_NO_TF"] = "1"
# chat_fixed.py
import numpy as np
import torch, requests
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words
from state_manager import StateManager
import threading
from dotenv import load_dotenv
from notion_client import Client
from typing import Optional, List, Dict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import socket
from datetime import datetime
import rapidfuzz
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
import os
import logging

# Ẩn bớt log của TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=full, 1=warning+, 2=error+, 3=fatal


# Tắt progress bar của HuggingFace Hub (tải model)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Giảm log của transformers & sentence-transformers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)


# ============== CẤU HÌNH ==============
ENV_PATH = r"D:\HTML\a\rag\.env"
try:
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH, override=True)
        # Sau load_dotenv:
except Exception:
    pass

print("=== DEBUG ENV CHECK ===")
print("ENV_PATH =", ENV_PATH, "| exists:", os.path.exists(ENV_PATH))
print("NOTION_API_KEY =", os.getenv("NOTION_API_KEY"))
print("NOTION_BASE_URL =", os.getenv("NOTION_BASE_URL"))
print("DATABASE_ID_FAQ =", os.getenv("DATABASE_ID_FAQ"))
print("========================")

_notion_cached = None
_notion_warned_once = False  # chỉ cảnh báo 1 lần khi lỗi HTTP push

# Ollama (có thể tắt nếu lỗi mạng)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2:1.5b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "60"))
ENABLE_OLLAMA_APPEND = os.getenv("ENABLE_OLLAMA_APPEND", "true").lower() != "false"
MAX_OLLAMA_APPEND_TOKENS = 150
print("[Ollama] URL:", OLLAMA_URL, "| model:", OLLAMA_MODEL, "| timeout:", OLLAMA_TIMEOUT)
FAQ_API_URL = "http://localhost:8000/search"
INVENTORY_API_URL = "http://localhost:8000/inventory"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_DB_PATH = os.path.join(BASE_DIR, "chat.db")
print(f"[ChatDB] Using: {CHAT_DB_PATH}")
DB_PATH = CHAT_DB_PATH

FAQ_DB_PATH = os.path.join(BASE_DIR, "faq.db")
CONF_THRESHOLD = 0.60
LOG_ALL_QUESTIONS = True

INTERRUPT_INTENTS = set()
CANCEL_WORDS = {"hủy", "huỷ", "huy", "cancel", "thoát", "dừng", "đổi chủ đề", "doi chu de"}
LAST_BOOK_CONTEXT = None

import unicodedata

def normalize_vi(text: str) -> str:
    text = (text or "").lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", text)
# ========== EMBEDDING MODEL ==========
try:
    embed_model = SentenceTransformer("keepitreal/vietnamese-sbert")
except Exception:
    embed_model = None  # fallback

# ============== DB helpers ==============
def ensure_main_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True) if os.path.dirname(DB_PATH) else None
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT,
            bot_reply   TEXT,
            intent_tag  TEXT,
            confidence  REAL,
            time        TEXT
        );
        """
    )
    conn.commit()
    return conn
def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def ensure_questions_log_db() -> None:
    dir_name = os.path.dirname(FAQ_DB_PATH)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    conn2 = sqlite3.connect(FAQ_DB_PATH)
    cur2 = conn2.cursor()
    cur2.execute(
        """
        CREATE TABLE IF NOT EXISTS questions_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question   TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            synced     INTEGER DEFAULT 0
        )
        """
    )
    conn2.commit()
    conn2.close()

def log_question_for_notion(question: str) -> None:
    if not question or not question.strip():
        return
    ensure_questions_log_db()
    conn2 = sqlite3.connect(FAQ_DB_PATH)
    cur2 = conn2.cursor()
    cur2.execute(
        "INSERT INTO questions_log (question, synced) VALUES (?, 0)",
        (question.strip(),),
    )
    conn2.commit()
    conn2.close()

# ============== Model load ==============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("intents.json", "r", encoding="utf-8-sig") as f:
    intents = json.load(f)

_data = torch.load("data.pth", map_location=device)
input_size  = _data["input_size"]
hidden_size = _data["hidden_size"]
output_size = _data["output_size"]
all_words   = _data["all_words"]
tags        = _data["tags"]
model_state = _data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

try:
    state_mgr = StateManager("flows.json")
except Exception:
    state_mgr = StateManager()




# ============== FAQ / Inventory ==============
def get_faq_response(sentence: str) -> Optional[str]:
    try:
        resp = requests.get(FAQ_API_URL, params={"q": sentence}, timeout=5)
        if resp.status_code != 200:
            print(f"[FAQ] HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        if not isinstance(data, list) or not data:
            return None
        lines: List[str] = []
        lines.append("📖 **Kết quả FAQ:**\n")
        lines.append("| Câu hỏi | Trả lời |")
        lines.append("|---------|---------|")
        for item in data:
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            if q or a:
                q = q.replace("|", "｜")
                a = a.replace("|", "｜")
                lines.append(f"| {q} | {a} |")
        return "\n".join(lines) if len(lines) > 3 else None
    except requests.RequestException as e:
        print(f"[FAQ] Lỗi kết nối API: {e}")
        return None
    except Exception as e:
        print(f"[FAQ] Lỗi xử lý dữ liệu: {e}")
        return None

def get_inventory_response(sentence: str) -> Optional[str]:
    try:
        resp = requests.get(INVENTORY_API_URL, params={"book_name": sentence}, timeout=5)
        if resp.status_code != 200:
            print(f"[Inventory] HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        if isinstance(data, list) and data:
            book = data[0]
            name = book.get("name")
            author = book.get("author", "?")
            year = book.get("year", "?")
            quantity = book.get("quantity", "?")
            status = book.get("status", "?")
            if name:
                return (
                    f"Sách '{name}' của tác giả {author}, năm xuất bản {year}, "
                    f"số lượng: {quantity}, trạng thái: {status}"
                )
        return None
    except requests.RequestException as e:
        print(f"[Inventory] Lỗi kết nối API: {e}")
        return None
    except Exception as e:
        print(f"[Inventory] Lỗi xử lý dữ liệu: {e}")
        return None



# ============== CORE chat ==============

def ollama_alive() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL.rstrip('/')}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False
        


def _dns_ok(host: str, timeout_s: float = 3.0) -> bool:
    try:
        socket.setdefaulttimeout(timeout_s)
        socket.getaddrinfo(host, 443)
        return True
    except Exception:
        return False
def pull_approved_from_notion_to_sqlite():
    token, dbid, mode, base = _resolve_notion_env()
    url = f"{base.rstrip('/')}/databases/{dbid}/query"
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": os.getenv("NOTION_VERSION", "2022-06-28"),
        "Content-Type": "application/json",
    }
    body = {
        "filter": {
            "and": [
                {"property": "Approved", "checkbox": {"equals": True}},
                # Nếu bạn tạo thêm cột "Synced" (checkbox) trong Notion:
                # {"property": "Synced", "checkbox": {"equals": False}},
            ]
        }
    }
    r = requests.post(url, headers=headers, json=body, timeout=12)
    r.raise_for_status()
    data = r.json()

    conn = ensure_main_db()
    cur = conn.cursor()

    for row in data.get("results", []):
        props = row.get("properties", {})
        q = props.get("Question", {}).get("rich_text", [{}])[0].get("plain_text", "")
        a = props.get("Answer", {}).get("rich_text", [{}])[0].get("plain_text", "")

        # lưu vào SQLite (ví dụ conversations hay bảng riêng)
        cur.execute(
            "INSERT INTO conversations(user_message, bot_reply, intent_tag, confidence, time) VALUES (?,?,?,?,?)",
            (q, a, None, 1.0, _now()),
        )
        # Đánh dấu đã sync nếu bạn có cột Synced trong Notion:
        # page_id = row["id"]
        # requests.patch(f"{base.rstrip('/')}/pages/{page_id}",
        #    headers={**headers, "Content-Type": "application/json"},
        #    json={"properties": {"Synced": {"checkbox": True}}})

    conn.commit()
    conn.close()

# ============== Notion helpers (ntn_ token, auto-mapping) ==============
from functools import lru_cache

def _resolve_notion_env():
    try:
        if os.path.exists(ENV_PATH):
            load_dotenv(ENV_PATH, override=True)
    except Exception:
        pass
    token = os.getenv("NOTION_TOKEN") or os.getenv("NOTION_API_KEY") or ""
    dbid  = (
        os.getenv("NOTION_DATABASE_ID")
        or os.getenv("DATABASE_ID_FAQ")
        or os.getenv("DATABASE_ID_BOOKS")
        or os.getenv("DATABASE_ID_MAJORS")
        or ""
    )
    base  = (os.getenv("NOTION_BASE_URL") or "https://api.notion.com/v1").rstrip("/")
    mode  = "sdk" if token.startswith("secret_") else "http"  # ntn_ => http

    # Fallback an toàn nếu đang trỏ tới ntn-api nhưng DNS/route hỏng
    if token.startswith("ntn_") and "ntn-api.notion.so" in base:
        if not _dns_ok("ntn-api.notion.so"):
            base = "https://api.notion.com/v1"

    return token, dbid, mode, base

def _rt(txt: str):
    return [{"type": "text", "text": {"content": txt or ""}}]

def _http_session_with_retry(total=2, backoff=0.6):
    s = requests.Session()
    retry = Retry(
        total=total,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST", "HEAD"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

def _ntn_session():
    # nhẹ hơn, ưu tiên giảm chờ
    return _http_session_with_retry(total=1, backoff=0.4)

def ntn_ok(base: str) -> bool:
    """Preflight: confirm CF/Notion phản hồi để tránh timeout kéo dài."""
    base = (base or "").rstrip("/")
    try:
        r = requests.get("https://api.notion.com/v1/status", timeout=6)
        if r.status_code not in (200, 400, 401, 405):
            print("[Preflight] api.notion.com status:", r.status_code)
    except requests.exceptions.RequestException:
        return False

    if "ntn-api.notion.so" in base:
        try:
            rr = requests.head(f"{base}/pages", timeout=6)
            return rr.status_code in (200,201,400,401,403,405,429,500,502,503,504,530)
        except requests.exceptions.RequestException:
            return False
    return True

def _http_create_page(token: str, base_url: str, payload: dict, timeout_s: float = 15.0):
    """POST /pages, trả (ok, status, body_text)."""
    url = f"{base_url.rstrip('/')}/pages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
        "Notion-Version": os.getenv("NOTION_VERSION", "2022-06-28"),
        "Host": "ntn-api.notion.so" if "ntn-api.notion.so" in base_url else "api.notion.com",
    }
    try:
        sess = _ntn_session()
        r = sess.post(url, headers=headers, json=payload, timeout=timeout_s, allow_redirects=True)
        ok = r.status_code in (200, 201)
        return ok, r.status_code, r.text
    except requests.exceptions.Timeout:
        return False, 408, "timeout"
    except Exception as e:
        return False, -1, f"{type(e).__name__}: {e}"

@lru_cache(maxsize=8)
def _fetch_db_schema(token: str, base: str, dbid: str) -> dict:
    """Lấy schema DB để auto-map properties (cache theo (token,base,dbid))."""
    url = f"{base.rstrip('/')}/databases/{dbid}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": os.getenv("NOTION_VERSION", "2022-06-28"),
        "Accept": "application/json",
    }
    sess = _http_session_with_retry(total=2, backoff=0.5)
    r = sess.get(url, headers=headers, timeout=10)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"GET /databases/{dbid} FAIL {r.status_code}: {r.text[:500]}")
    return r.json()

def _pick_prop_by_type(props: dict, want_type: str, prefer_names: list[str]) -> Optional[str]:
    """Chọn tên cột theo type: ưu tiên theo danh sách tên gợi ý, fallback cột bất kỳ cùng type."""
    # ưu tiên theo tên
    lower_props = {k.lower(): k for k in props.keys()}
    for name in prefer_names:
        key = lower_props.get(name.lower())
        if key and props.get(key, {}).get("type") == want_type:
            return key
    # fallback: lấy cột đầu tiên có type phù hợp
    for k, v in props.items():
        if v.get("type") == want_type:
            return k
    return None

def _ensure_select_option(token: str, base: str, dbid: str, prop_name: str, option_name: str) -> str:
    """
    Đảm bảo option select tồn tại; nếu chưa có sẽ thêm (best effort).
    Trả lại option_name (có thể đã tồn tại hoặc vừa tạo).
    """
    # Đọc schema
    schema = _fetch_db_schema(token, base, dbid)
    props = schema.get("properties", {})
    prop = props.get(prop_name, {})
    if prop.get("type") != "select":
        return option_name  # không phải select thì bỏ qua

    options = prop.get("select", {}).get("options", []) or []
    names = {opt.get("name"): opt.get("id") for opt in options if isinstance(opt, dict)}
    if option_name in names:
        return option_name

    # Thử thêm option qua update database
    url = f"{base.rstrip('/')}/databases/{dbid}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Notion-Version": os.getenv("NOTION_VERSION", "2022-06-28"),
    }
    new_opt = {"name": option_name}
    body = {
        "properties": {
            prop_name: {
                "select": {
                    "options": options + [new_opt]
                }
            }
        }
    }
    try:
        r = requests.patch(url, headers=headers, json=body, timeout=12)
        if r.status_code in (200, 201):
            return option_name
        else:
            # Không tạo được option → vẫn dùng tên option (Notion sẽ reject nếu chưa có)
            print(f"[Notion] WARN: add select option FAIL {r.status_code}: {r.text[:400]}")
            return option_name
    except Exception as e:
        print(f"[Notion] WARN: add select option error: {e}")
        return option_name


def _build_dynamic_payload_force(dbid: str, q: str, a: str) -> dict:
    title_txt = (q or "Câu hỏi").strip()[:200]
    today_iso = datetime.now().date().isoformat()

    props = {
        "Question": {"rich_text": [{"type": "text", "text": {"content": q or ""}}]},
        "Answer":   {"rich_text": [{"type": "text", "text": {"content": a or ""}}]},
        # Cho item xuất hiện ngay ở view chính:
        "Approved": {"checkbox": True},  # <-- bật nếu view đang lọc Approved = checked
        "Language": {"select": {"name": "Tiếng Việt"}},  # <-- khớp filter Language
        "Last Update": {"date": {"start": today_iso}},
    }

    # Nếu bảng của bạn BẮT BUỘC có Category để vào view, set thêm 1 value hợp lệ:
    # props["Category"] = {"select": {"name": "Quy định"}}

    return {
        "parent": {"database_id": dbid},
        "properties": props,
    }




def push_to_notion(q: str, a: str):
    """
    Đẩy ngay từng dòng lên Notion (ntn_). Tự dò schema và map properties.
    In lỗi chi tiết khi fail để bạn sửa đúng chỗ.
    """
    global _notion_warned_once
    q = (q or "").strip(); a = (a or "").strip()
    if not q:
        return

    token, dbid, mode, base = _resolve_notion_env()
    if not token or not dbid:
        print("[Notion] Bỏ qua: thiếu token/dbid.")
        return

    # Chỉ hỗ trợ http (ntn_) ở đây; nếu bạn dùng secret_, có thể nhánh SDK.
    if mode != "http":
        print("[Notion] Bạn đang dùng secret_; nhánh HTTP này dành cho ntn_.")
        return

    # Preflight – tránh đợi timeout vô ích
    # 👉 Preflight: có thể BỎ QUA nếu FORCE_PUSH_NOTION=1
    force_push = os.getenv("FORCE_PUSH_NOTION", "0") == "1"
    if not force_push and not ntn_ok(base):
        if not _notion_warned_once:
            print("[Notion] Gateway hiện không reachable → bỏ qua lần này.")
            _notion_warned_once = True
        return
    else:
        if force_push:
            print("[Notion] FORCE: bỏ qua preflight, thử push trực tiếp...")


    # Build payload theo schema thực tế
    try:
        payload = _build_dynamic_payload_force(dbid, q, a)

    except Exception as e:
        print(f"[Notion] Build payload error: {e}")
        return

    ok, status, body = _http_create_page(token, base, payload, timeout_s=15.0)
    if ok:
        print(f"[Notion] OK ({status})")
    else:
        # In body đầy đủ để thấy lỗi thật (property nào sai type/tên/option)
        print(f"[Notion] FAIL ({status})\n{body[:2000]}")


def _ntn_session():
    s = requests.Session()
    retry = Retry(
        total=1,               # chỉ 1 lần retry nhẹ để không chờ lâu
        backoff_factor=0.4,
        status_forcelist=[429,500,502,503,504],
        allowed_methods=["POST", "HEAD"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s
# ============== Ollama append (an toàn) ==============
def sanitize_vi(extra: str) -> str:
    if not extra: return ""
    extra = re.sub(r'[\u3400-\u9FFF\uF900-\uFAFF]+', '', extra)
    extra = re.sub(r'[\U0001F300-\U0001FAFF]', '', extra)
    extra = extra.replace('“','').replace('”','').replace('"','').strip()
    extra = re.sub(r'\s+', ' ', extra)
    banned_starts = ("chào mừng", "rất tiếc", "xin chào", "cảm ơn")
    if extra.lower().startswith(banned_starts): return ""
    if len(extra.split()) < 3: return ""
    return extra
def get_recent_history(limit=6):
    """Lấy luân phiên Q/A gần nhất, mới → cũ (tối đa limit dòng)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            SELECT user_message, bot_reply, time
            FROM conversations
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        conn.close()
        # đảo lại cho thành cũ → mới
        rows.reverse()
        return rows
    except Exception:
        return []

def ollama_generate_continuation(base_reply: str, user_message: str, max_sentences=3) -> str:
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    history = get_recent_history(limit=8)

    # Ghép lịch sử: Q/A ngắn gọn
    hist_lines = []
    for q, a, t in history:
        q = (q or "").strip()
        a = (a or "").strip()
        if q or a:
            hist_lines.append(f"- User: {q}")
            hist_lines.append(f"  Bot: {a}")
    hist_block = "\n".join(hist_lines[-14:])  # tránh dài quá

    system_prompt = (
        "Bạn là trợ lý thư viện DHTN. Dựa vào lịch sử hội thoại dưới đây, "
        "hãy VIẾT TIẾP phần trả lời cho mượt mà, chỉ thêm ý bổ sung hợp lý, "
        "KHÔNG lặp lại nguyên văn, KHÔNG mở chủ đề mới, KHÔNG bịa số liệu. "
        "Nếu lịch sử không giúp ích, trả về chuỗi RỖNG.\n"
        "Giới hạn 1–3 câu ngắn. Chỉ tiếng Việt."
    )

    user_prompt = (
        f"Lịch sử gần đây:\n{hist_block}\n\n"
        f"Câu trả lời hiện tại của bot:\n{base_reply}\n\n"
        f"Người dùng vừa hỏi:\n{user_message}\n\n"
        f"YÊU CẦU: Viết tiếp ngắn gọn (1–3 câu) bổ sung ý dựa trên lịch sử. "
        f"Nếu không phù hợp, trả về rỗng."
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{system_prompt}\n\n{user_prompt}",
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "repeat_penalty": 1.15,
            "num_predict": 120
        }
    }

    try:
        r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        if r.status_code != 200:
            print(f"[Ollama-continue] HTTP {r.status_code}: {r.text[:200]}")
            return ""
        extra = (r.json().get("response") or "").strip()
        # làm sạch ngắn gọn
        extra = re.sub(r'\s+', ' ', extra)
        if not extra or extra.lower() in ("", "rỗng", "(rỗng)"):
            return ""
        # cắt tối đa 3 câu
        sentences = [s.strip() for s in re.split(r'[.!?…]+', extra) if s.strip()]
        extra_short = ". ".join(sentences[:max_sentences]).strip()
        return (extra_short + ".") if extra_short and not extra_short.endswith(".") else extra_short
    except Exception as e:
        print("[Ollama-continue] Error:", e)
        return ""
    # tự động lấy intent từ notion 
def get_all_categories():
    conn = sqlite3.connect(FAQ_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT category FROM faq WHERE category IS NOT NULL")
    rows = cur.fetchall()
    conn.close()
    # trả về list tên categories, loại bỏ None, rỗng
    cats = [ (r[0] or "").strip() for r in rows ]
    cats = [c for c in cats if c]
    cats.extend(["Thông tin ngành", "Tra cứu sách"])
    return sorted(set(cats))
def get_all_major_names():
    conn = sqlite3.connect(FAQ_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name FROM majors")
    rows = cur.fetchall()
    conn.close()
    return [r[0].strip().lower() for r in rows]

def answer_from_majors(user_message: str) -> str:
    try:
        # --- 1. Trích tên ngành ---
        extract_prompt = f"""
Bạn là trợ lý thư viện.
Hãy trích tên NGÀNH từ câu hỏi sau.
Nếu không tìm thấy ngành → trả về rỗng.

Câu hỏi: "{user_message}"

Chỉ trả về tên ngành (vd: Công nghệ thông tin, Kinh tế, CNTT).
Không giải thích thêm.
"""
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": extract_prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 50}
        }
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT)

        major_key = (r.json().get("response") or "").strip().split("\n")[0]
        major_key = re.sub(r'[^0-9a-zA-ZÀ-ỹ\s]', '', major_key)

        if not major_key:
            return "Mình chưa xác định được ngành trong câu hỏi."

        # --- 2. Tìm ngành trong bảng majors ---
        conn = sqlite3.connect(FAQ_DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            SELECT name, major_id, description
            FROM majors
            WHERE name LIKE ? OR major_id LIKE ?
        """, (f"%{major_key}%", f"%{major_key}%"))

        rows = cur.fetchall()
        conn.close()

        if not rows:
            return f"Không tìm thấy ngành liên quan: {major_key}"

        # format
        text = "\n".join(f"- {name} (Mã: {mid}): {desc}" for name, mid, desc in rows)

        # --- 3. Viết câu trả lời ---
        answer_prompt = f"""
Người dùng hỏi: "{user_message}"
Dưới đây là thông tin ngành tìm được:

{text}

Hãy trả lời tự nhiên, KHÔNG bịa thêm.
"""
        payload2 = {
            "model": OLLAMA_MODEL,
            "prompt": answer_prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 150}
        }
        rr = requests.post(f"{OLLAMA_URL}/api/generate", json=payload2, timeout=OLLAMA_TIMEOUT)
        return (rr.json().get("response") or "").strip()

    except Exception as e:
        return f"[LỖI majors] {e}"


def _llm_format_books_answer(question: str, books: list[tuple], mode: str, extra_label: str = "") -> str:
    """
    Dùng Ollama để viết câu trả lời cho đẹp, NHƯNG chỉ dựa trên list `books`.
    books: list tuple (name, author, year, quantity, status, major_name)
    mode: 'book' | 'author' | 'major' | 'list'
    extra_label: tên tác giả / tên ngành / tên sách gốc nếu muốn nhắc lại
    """
    if not ollama_alive() or not books:
        return ""  # để answer_from_books fallback sang format cứng

    # Ghép block sách gửi cho LLM
    lines = []
    for idx, (name, author, year, qty, status, major_name) in enumerate(books, start=1):
        major_name = major_name or "Không rõ"
        lines.append(
            f"{idx}) Tên: {name} | Tác giả: {author} | Năm: {year} | "
            f"SL: {qty} | Trạng thái: {status} | Ngành: {major_name}"
        )
    books_block = "\n".join(lines)

    if mode == "book":
        mode_desc = "MỘT cuốn sách cụ thể mà người dùng đang hỏi."
    elif mode == "author":
        mode_desc = f"các sách của TÁC GIẢ {extra_label}."
    elif mode == "major":
        mode_desc = f"các sách thuộc NGÀNH {extra_label}."
    else:  # 'list'
        mode_desc = "DANH SÁCH các sách liên quan đến câu hỏi của người dùng."

    system_prompt = f"""
Bạn là trợ lý thư viện. Bạn sẽ được cung cấp:
- CÂU HỎI của người dùng.
- DANH SÁCH SÁCH lấy trực tiếp từ cơ sở dữ liệu thư viện.

NHIỆM VỤ:
1. Dựa vào DANH SÁCH SÁCH bên dưới để trả lời câu hỏi của người dùng về {mode_desc}.
2. CHỈ ĐƯỢC SỬ DỤNG những sách xuất hiện trong danh sách bên dưới.
   **KHÔNG ĐƯỢC BỊA THÊM tên sách, tác giả, năm, trạng thái, số lượng, ngành mới.**
3. Nếu danh sách chỉ có 1 sách → mô tả chi tiết chính cuốn đó.
4. Nếu câu hỏi là về NGÀNH → CHỈ ĐƯỢC liệt kê các sách thuộc ngành đó trong danh sách (không tự tạo thêm).
5. Trả lời bằng tiếng Việt, tự nhiên, ngắn gọn, dễ hiểu.

TUYỆT ĐỐI KHÔNG ĐƯỢC:
- Bịa thêm bất kỳ cuốn sách nào không có trong danh sách.
- Tự tạo tên sách, tác giả, năm xuất bản.
- Tự tạo thêm nội dung mô tả sách nếu danh sách không cung cấp.
- Gộp nhóm, thêm sách ví dụ minh hoạ ngoài danh sách.
- Đưa ra gợi ý không có trong dữ liệu.

NẾU DANH SÁCH CHỈ CÓ 1 CUỐN → chỉ trả đúng cuốn đó.
NẾU DANH SÁCH CÓ NHIỀU CUỐN → CHỈ LIỆT KÊ NHỮNG CUỐN ĐÃ CHO.
KHÔNG BAO GIỜ LIỆT KÊ THÊM 3–7 CUỐN KHÁC TỰ NGHĨ RA.
"""

    user_prompt = f"""
Câu hỏi người dùng: "{question}"

DANH SÁCH SÁCH TỪ CƠ SỞ DỮ LIỆU:
{books_block}

Hãy trả lời, NHỚ: chỉ dùng thông tin trong danh sách trên.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": system_prompt + "\n\n" + user_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,   # giảm max bịa
            "num_predict": 400
        }
    }

    try:
        r = requests.post(f"{OLLAMA_URL.rstrip('/')}/api/generate",
                          json=payload, timeout=OLLAMA_TIMEOUT)
        if r.status_code != 200:
            print("[books-llm-format] HTTP", r.status_code, r.text[:200])
            return ""
        resp = (r.json().get("response") or "").strip()
        return resp
    except Exception as e:
        print("[books-llm-format] Error:", e)
        return ""

MAJOR_EMB = []       # danh sách vector
MAJOR_META = []      # (name, major_id, description)
def vector(txt: str):
    if not embed_model:
        return None
    return embed_model.encode(txt, normalize_embeddings=True)
def build_major_embedding_index():
    global MAJOR_EMB, MAJOR_META

    conn = sqlite3.connect(FAQ_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name, major_id, description FROM majors")
    rows = cur.fetchall()
    conn.close()

    MAJOR_META = rows
    MAJOR_EMB = [vector(r[0]) for r in rows]

def search_majors_by_embedding(query: str, top_k=1):
    if not embed_model or not MAJOR_EMB:
        return []
    qv = vector(query)
    sims = np.dot(MAJOR_EMB, qv)
    idx = np.argsort(sims)[::-1][:top_k]
    return [(i, sims[i], MAJOR_META[i]) for i in idx]

# ====== EMBEDDING CHO BOOKS (SEMANTIC SEARCH) ======
EMB_MODEL_NAME_BOOKS = os.getenv(
    "BOOK_EMB_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # hoặc model khác bạn thích
)
book_emb_model = SentenceTransformer(EMB_MODEL_NAME_BOOKS)
# Model embedding cho majors
MAJOR_EMB_MODEL = os.getenv(
    "MAJOR_EMB_MODEL",
    "keepitreal/vietnamese-sbert"
)
major_emb_model = SentenceTransformer(MAJOR_EMB_MODEL)
print("[Books-Emb] Loading SentenceTransformer model:", EMB_MODEL_NAME_BOOKS)
book_emb_model = SentenceTransformer(EMB_MODEL_NAME_BOOKS)

# Cache: embeddings + dữ liệu thô của books
BOOK_EMBS: np.ndarray | None = None   # shape (N_books, dim)
BOOK_ROWS: list[tuple] | None = None  # [(name, author, year, qty, status, major_name), ...]


def build_book_embedding_index() -> tuple[np.ndarray, list[tuple]]:
    """
    Đọc toàn bộ bảng books + majors và build index embedding cho SÁCH.
    Chỉ build 1 lần, sau đó dùng lại từ cache.
    """
    global BOOK_EMBS, BOOK_ROWS

    # Nếu đã build trước đó rồi thì dùng lại
    if BOOK_EMBS is not None and BOOK_ROWS is not None:
        return BOOK_EMBS, BOOK_ROWS

    conn = sqlite3.connect(FAQ_DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT b.name, b.author, b.year, b.quantity, b.status, m.name
        FROM books b
        LEFT JOIN majors m ON b.major_id = m.major_id
    """)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        dim = book_emb_model.get_sentence_embedding_dimension()
        BOOK_EMBS = np.zeros((0, dim), dtype=np.float32)
        BOOK_ROWS = []
        return BOOK_EMBS, BOOK_ROWS

    # Chuẩn bị text mô tả mỗi cuốn sách để embedding
    texts = []
    for (name, author, year, qty, status, major_name) in rows:
        name = name or ""
        author = author or ""
        major_name = major_name or ""
        year = str(year or "")
        status = status or ""
        t = (
            f"Sách: {name}. Tác giả: {author}. Ngành: {major_name}. "
            f"Năm: {year}. Chủ đề: {name} {major_name} {author}"
        )
        texts.append(t)

    print(f"[Books-Emb] Building embeddings cho {len(texts)} sách...")
    emb = book_emb_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # để cosine = dot
        show_progress_bar=False,
    )

    BOOK_EMBS = emb
    BOOK_ROWS = rows
    print("[Books-Emb] Done.")
    return BOOK_EMBS, BOOK_ROWS


def search_books_by_embedding(
    query: str,
    top_k: int = 10,
    min_sim: float = 0.45,
) -> list[tuple[tuple, float]]:
    """
    Tìm sách theo NGHĨA bằng cosine similarity.
    Trả về list[(row, sim)] đã sort giảm dần.
    row đúng cấu trúc:
        (name, author, year, quantity, status, major_name)
    """
    emb, rows = build_book_embedding_index()
    if emb.shape[0] == 0:
        return []

    q_vec = book_emb_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    sims = emb @ q_vec  # cosine vì đã normalize
    idx_sorted = np.argsort(-sims)

    results: list[tuple[tuple, float]] = []
    for i in idx_sorted[:top_k]:
        sim = float(sims[i])
        if sim < min_sim:
            continue
        results.append((rows[i], sim))
    return results


def answer_from_books(user_message: str) -> str:
    """
    Tra cứu sách dựa trên EMBEDDING (semantic search),
    sau đó (nếu được) nhờ Ollama viết lại câu trả lời cho tự nhiên hơn.

    Không dùng keyword, không dùng fuzzy cho BOOK nữa.
    """
    try:
        text_raw = (user_message or "").strip()
        if not text_raw:
            return "Mình chưa nhận được nội dung để tra cứu sách."

        # Lấy top sách theo NGHĨA
        results = search_books_by_embedding(text_raw, top_k=12, min_sim=0.45)
        if not results:
            return (
                "Hiện mình chưa tìm được sách phù hợp trong danh mục. "
                "Bạn thử ghi rõ hơn tên sách, tác giả hoặc ngành nhé."
            )

        # Tách rows & similarity
        rows = [r[0] for r in results]
        sims = [r[1] for r in results]

        # Xem câu hỏi có dạng "liệt kê / tất cả" không
        text_norm = normalize_vi(text_raw)
        list_keywords = [
            "tat ca", "tất cả",
            "liet ke", "liệt kê",
            "danh sach", "danh sách",
            "sach lien quan", "sách liên quan",
            "cac sach", "các sách",
            "nhung sach", "những sách",
        ]
        is_list_query = any(k in text_norm for k in list_keywords)

        # Nếu câu hỏi kiểu liệt kê → đưa list cho LLM
        if is_list_query or len(rows) > 3:
            books = rows  # list[(name, author, year, qty, status, major_name)]
            llm_ans = _llm_format_books_answer(
                text_raw,
                books,
                mode="list",
            )
            if llm_ans:
                return llm_ans

            # Fallback: liệt kê cứng
            block = "\n".join(
                f"- {n} – {a}, {y}, SL: {q}, Trạng thái: {s}, Ngành: {mj or 'Không rõ'}"
                for (n, a, y, q, s, mj) in books
            )
            return f"Dưới đây là một số sách liên quan đến câu hỏi của bạn:\n\n{block}"

        # Ngược lại: coi là hỏi 1 CUỐN SÁCH GẦN NHẤT
        best_row = rows[0]
        best_sim = sims[0]
        global LAST_BOOK_CONTEXT
        LAST_BOOK_CONTEXT = best_row

        if best_sim < 0.5:
            return (
                "Mình chưa chắc sách nào phù hợp với câu hỏi này. "
                "Bạn thử nêu rõ tên sách, tác giả hoặc mô tả chi tiết hơn nhé."
            )

        n, a, y, q, s, mj = best_row
        major_label = mj or "Không rõ"

        # Cho LLM format đẹp hơn (mode 'book')
        llm_ans = _llm_format_books_answer(
            text_raw,
            [best_row],
            mode="book",
            extra_label=n,
        )
        if llm_ans:
            return llm_ans

        # Fallback: format cứng
        return (
            f"**Thông tin sách gần nhất với câu hỏi của bạn:**\n"
            f"- Tên: {n}\n"
            f"- Tác giả: {a}\n"
            f"- Năm XB: {y}\n"
            f"- Số lượng: {q}\n"
            f"- Trạng thái: {s}\n"
            f"- Ngành: {major_label}"
        )

    except Exception as e:
        return f"[LỖI books-emb] {e}"




def classify_category(user_message: str) -> str:
    """
    Phân loại intent dùng LLM.
    Ưu tiên LLM → fallback rule nhẹ nếu LLM trả linh tinh.
    KHÔNG fuzzy, KHÔNG rule ép cứng như trước.
    """

    msg = (user_message or "").strip()
    if not msg:
        return "Tra cứu sách"

    # ===== 1. Lấy category thật trong FAQ =====
    try:
        conn = sqlite3.connect(FAQ_DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT category FROM faq WHERE category IS NOT NULL")
        rows = cur.fetchall()
        conn.close()
        faq_categories = [(r[0] or "").strip() for r in rows if r[0]]
    except:
        faq_categories = []

    faq_categories = sorted(set([c for c in faq_categories if c]))

    # Category cố định
    special = ["Tra cứu sách", "Thông tin ngành"]
    allowed = special + faq_categories

    def norm(s: str) -> str:
        return normalize_vi((s or "").strip())

    categories_list_str = "\n".join(f"- {c}" for c in allowed)

    system_prompt = f"""
Bạn là trợ lý thư viện.
Nhiệm vụ: phân loại câu hỏi vào ĐÚNG MỘT category trong danh sách sau:

{categories_list_str}

QUY TẮC:
- Về sách → "Tra cứu sách".
- Về ngành học → "Thông tin ngành".
- Về quy định, thủ tục, nhiệm vụ, chức năng, giờ mở cửa, nội quy → chọn đúng category trong FAQ.
- KHÔNG bịa thêm category mới.
- Chỉ được trả về đúng 1 category, không thêm chữ nào.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": system_prompt + "\n\nCâu hỏi: " + msg,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 32},
    }

    # ===== 2. Gọi LLM =====
    try:
        r = requests.post(
            f"{OLLAMA_URL.rstrip('/')}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT
        )
        raw = (r.json().get("response") or "").strip().splitlines()[0]
        c = raw.strip().lstrip("-•* ").strip('"').strip("'")
        c_norm = norm(c)

        # Nếu LLM trả đúng → OK
        for cat in allowed:
            if c_norm == norm(cat):
                return cat

    except Exception as e:
        print("[classify_category] LLM error:", e)

    # ===== 3. Fallback rule (NHẸ, KHÔNG ép sai FAQ) =====
    msg_n = norm(msg)

    # hỏi ngành
    if any(k in msg_n for k in ["nganh", "chuyen nganh", "ma nganh", "hoc nganh"]):
        return "Thông tin ngành"

    # hỏi sách
    if any(k in msg_n for k in ["sach", "giáo trình", "giao trinh", "tai lieu"]):
        return "Tra cứu sách"

    # cuối cùng → cho về sách (an toàn nhất)
    return "Tra cứu sách"



def detect_book_followup_intent(user_message: str) -> str:
    """
    Dùng LLM để hiểu câu hỏi tiếp theo đang hỏi gì về cuốn sách trong LAST_BOOK_CONTEXT.
    Trả về 1 trong:
    - 'quantity' : hỏi về số lượng, còn nhiều không, còn bao nhiêu quyển, v.v.
    - 'status'   : hỏi kiểu còn hàng không, tình trạng ra sao, có sẵn không,...
    - 'other'    : hỏi cái khác nhưng vẫn liên quan cuốn sách (vd: nội dung, khó/dễ,...)
    - 'none'     : không liên quan tới cuốn sách trước.
    """
    global LAST_BOOK_CONTEXT
    if not ollama_alive() or LAST_BOOK_CONTEXT is None:
        return "none"

    n, a, y, q, s, mj = LAST_BOOK_CONTEXT
    book_info = (
        f"Tên: {n}. Tác giả: {a}. Năm: {y}. "
        f"Số lượng: {q}. Trạng thái: {s}. Ngành: {mj or 'Không rõ'}."
    )

    system_prompt = """
Bạn là trợ lý thư viện.
Bạn sẽ nhận được:
- Thông tin một cuốn sách.
- Câu hỏi mới của người dùng (sau khi họ vừa hỏi về cuốn sách này).

NHIỆM VỤ:
Hiểu ngữ nghĩa câu hỏi mới và phân loại nó thành đúng 1 nhãn sau:

- "quantity"  → nếu người dùng hỏi về số LƯỢNG, còn bao nhiêu quyển, còn nhiều không, hết chưa,...
- "status"    → nếu người dùng hỏi về TÌNH TRẠNG / CÒN HÀNG KHÔNG, có sẵn để mượn không,...
- "other"     → nếu người dùng hỏi thứ khác nhưng vẫn LIÊN QUAN cuốn sách (nội dung, độ khó, nên học,...).
- "none"      → nếu câu hỏi KHÔNG liên quan tới cuốn sách trước.

Chỉ được trả về DUY NHẤT một từ trong 4 từ sau:
quantity
status
other
none
"""

    user_prompt = f"""
Thông tin sách:
{book_info}

Câu hỏi mới của người dùng: "{user_message}"

Hãy trả về đúng 1 từ trong: quantity, status, other, none.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": system_prompt + "\n\n" + user_prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 10},
    }

    try:
        r = requests.post(f"{OLLAMA_URL.rstrip('/')}/api/generate",
                          json=payload, timeout=OLLAMA_TIMEOUT)
        raw = (r.json().get("response") or "").strip().splitlines()[0].strip().lower()
        if raw in ("quantity", "status", "other", "none"):
            return raw
        return "none"
    except Exception as e:
        print("[followup-intent] error:", e)
        return "none"
def process_message(sentence: str) -> str:
    sentence = (sentence or "").strip()
    if not sentence:
        return "Xin lỗi, mình chưa hiểu ý bạn."

    reply: Optional[str] = None
    tag_to_log: Optional[str] = None
    confidence: float = 0.0
    text_norm = normalize_vi(sentence)

    global LAST_BOOK_CONTEXT

    # ====== BƯỚC 1: xử lý câu hỏi tiếp theo về CUỐN SÁCH trước đó ======
    if LAST_BOOK_CONTEXT is not None and ollama_alive():
        intent = detect_book_followup_intent(sentence)  # quantity | status | other | none

        if intent in ("quantity", "status", "other"):
            n, a, y, q, s, mj = LAST_BOOK_CONTEXT
            major_label = mj or "Không rõ"

            try:
                qty = int(q)
            except Exception:
                qty = None

            if intent == "quantity":
                if qty is None:
                    reply = (
                        f"Mình chưa có dữ liệu chính xác về số lượng sách **{n}**.\n"
                        f"- Trạng thái hiện tại: {s}."
                    )
                else:
                    if qty > 0:
                        reply = (
                            f"Sách **{n}** của {a} hiện trong hệ thống còn khoảng {qty} quyển.\n"
                            f"Trạng thái: {s}."
                        )
                    else:
                        reply = f"Sách **{n}** của {a} hiện đã hết hàng trong hệ thống."

            elif intent == "status":
                if qty is not None and qty > 0:
                    reply = (
                        f"Sách **{n}** của {a} hiện đang còn trong thư viện "
                        f"(khoảng {qty} quyển). Trạng thái: {s}."
                    )
                else:
                    reply = (
                        f"Sách **{n}** của {a} hiện không còn sẵn trong kho hoặc số lượng rất ít.\n"
                        f"Trạng thái ghi nhận: {s}."
                    )

            elif intent == "other":
                reply = (
                    f"Bạn đang hỏi thêm về sách **{n}** của {a} (năm {y}, ngành {major_label}).\n"
                    f"Hiện hệ thống chỉ lưu thông tin cơ bản: số lượng = {q}, trạng thái = {s}. "
                    f"Nếu bạn cần nội dung chi tiết, bạn có thể tra cứu sách trực tiếp tại thư viện."
                )

            if reply:
                tag_to_log = "Tra cứu sách (followup)"

    # ====== BƯỚC 2: Router chính ======
    if reply is None:
        if ollama_alive():
            # ----- 2A. Dùng LLM phân loại trước -----
            try:
                category = classify_category(sentence)
            except Exception as e:
                print("[process_message] classify_category error:", e)
                category = None

            if category:
                tag_to_log = category

            # 2A.1. Nếu LLM nói đây là câu hỏi về NGÀNH
            if category == "Thông tin ngành":
                reply = answer_from_majors(sentence)
                tag_to_log = "Thông tin ngành"

            # 2A.2. Nếu LLM nói đây là câu hỏi về SÁCH
            elif category == "Tra cứu sách":
                reply = answer_from_books(sentence)
                tag_to_log = "Tra cứu sách"

            # 2A.3. Còn lại: xem như FAQ (Quy định, Nhiệm vụ, Chức năng, ...)
            # 2A.3. Xử lý FAQ — không bịa, chỉ dùng dữ liệu SQLite
            elif category and category not in ("Tra cứu sách", "Thông tin ngành"):
                try:
                    conn_faq = sqlite3.connect(FAQ_DB_PATH)
                    cur = conn_faq.cursor()
                    cur.execute("""
                        SELECT question, answer
                        FROM faq
                        WHERE category = ?
                        AND (approved = 1 OR approved IS NULL)
                    """, (category,))
                    rows = cur.fetchall()
                    conn_faq.close()
                except Exception as e:
                    print("[FAQ SELECT error]", e)
                    rows = []

                if rows:
                    # Ghép block
                    faq_block = "\n\n".join(
                        f"{idx}) Q: {(q or '').strip()}\n   A: {(a or '').strip()}"
                        for idx, (q, a) in enumerate(rows, 1)
                    )

                    answer_prompt = f"""
            Bạn là trợ lý thư viện.
            Chỉ được dùng NỘI DUNG có trong danh sách Answer dưới đây.
            KHÔNG ĐƯỢC bịa thêm thông tin mới.
            KHÔNG được đưa ví dụ không nằm trong danh sách.
            Nếu không có Answer phù hợp → phải trả lời:
            "Hiện tại mình chưa có thông tin chính xác trong hệ thống thư viện về câu hỏi này."

            Câu hỏi của người dùng:
            {sentence}

            Danh sách Answer theo category "{category}":

            {faq_block}

            Hãy trả lời đúng nội dung, KHÔNG mở rộng ra ngoài.
            """

                    try:
                        payload = {
                            "model": OLLAMA_MODEL,
                            "prompt": answer_prompt,
                            "stream": False,
                            "options": {"temperature": 0.1, "num_predict": 200}
                        }
                        r = requests.post(
                            f"{OLLAMA_URL.rstrip('/')}/api/generate",
                            json=payload,
                            timeout=OLLAMA_TIMEOUT
                        )
                        if r.status_code == 200:
                            raw = (r.json().get("response") or "").strip()
                            # Nếu LLM bịa ngoài dữ liệu → phát hiện và chặn lại
                            if not raw or any(x in raw.lower() for x in [
                                "ví dụ", "ví du", "example", "theo mình", "mình nghĩ"
                            ]):
                                reply = ("Hiện tại mình chưa có thông tin chính xác "
                                        "trong hệ thống thư viện về câu hỏi này.")
                            else:
                                reply = raw
                            confidence = 0.9
                    except Exception as e:
                        print("[FAQ LLM error]", e)

                # không tìm được -> fallback
                if reply is None or not reply.strip():
                    reply = ("Hiện tại mình chưa có thông tin chính xác "
                            "trong hệ thống thư viện về câu hỏi này.")
                    confidence = 0.5

            # ----- 2B. Nếu LLM/FAQ không cho được câu trả lời → fallback embedding như cũ -----

                if book_hits:
                    reply = answer_from_books(sentence)
                    tag_to_log = tag_to_log or "Tra cứu sách"

                # majors embedding
                if reply is None or not reply.strip():
                    try:
                        major_hits = search_majors_by_embedding(sentence, top_k=1)
                    except Exception as e:
                        print("[process_message] major-emb error:", e)
                        major_hits = []

                    if major_hits and major_hits[0][1] >= 0.55:
                        reply = answer_from_majors(sentence)
                        tag_to_log = tag_to_log or "Thông tin ngành"

        else:
            # ====== Ollama không sống → fallback thuần embedding (không FAQ) ======
            try:
                book_hits = search_books_by_embedding(sentence, top_k=1, min_sim=0.55)
            except Exception as e:
                print("[process_message] book-emb (no LLM) error:", e)
                book_hits = []

            if book_hits:
                reply = answer_from_books(sentence)
                tag_to_log = "Tra cứu sách"
            else:
                try:
                    major_hits = search_majors_by_embedding(sentence, top_k=1)
                except Exception as e:
                    print("[process_message] major-emb (no LLM) error:", e)
                    major_hits = []

                if major_hits and major_hits[0][1] >= 0.55:
                    reply = answer_from_majors(sentence)
                    tag_to_log = "Thông tin ngành"

    # ====== Fallback nếu vẫn chưa có câu trả lời ======
    if reply is None or not reply.strip():
        reply = "Xin lỗi, mình chưa hiểu ý bạn."
        confidence = 0.0

    # ====== Append thêm cho mượt (nếu bật) ======
    if ENABLE_OLLAMA_APPEND and reply.strip() and ollama_alive():
        extra = ollama_generate_continuation(reply, sentence, max_sentences=3)
        if extra:
            reply = f"{reply.strip()} {extra.strip()}"

    # ====== Ghi log vào conversations ======
    conn = ensure_main_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO conversations(user_message, bot_reply, intent_tag, confidence, time) "
        "VALUES (?,?,?,?,?)",
        (sentence, reply, tag_to_log, confidence, _now()),
    )
    conn.commit()
    conn.close()

    # ====== Ghi thêm vào faq.db (inbox) ======
    try:
        log_question_for_notion(f"User: {sentence}\nBot: {reply}")
    except Exception as e:
        print(f"[Notion inbox] Lỗi ghi faq.db: {e}")

    # ====== Đẩy lên Notion (background) ======
    should_push = (
        LOG_ALL_QUESTIONS
        or reply.strip().startswith("Xin lỗi, mình chưa hiểu")
        or confidence < CONF_THRESHOLD
        or tag_to_log is None
    )
    if should_push:
        try:
            threading.Thread(target=push_to_notion, args=(sentence, reply), daemon=True).start()
        except Exception as e:
            print("Notion push error:", e)

    return reply



# ============== CLI ==============
def _test_push_notion_once():
    token, dbid, mode, base = _resolve_notion_env()
    tok_prefix = (token.split("_",1)[0]+"_") if "_" in token else token[:6]
    print("[TEST] mode:", mode, "| dbid:", dbid, "| base:", base, "| token_prefix:", tok_prefix)

    # Test /status (Cloudflare/Notion)
    try:
        r = requests.get("https://api.notion.com/v1/status", timeout=6)
        print("[TEST] status api.notion.com:", r.status_code)
    except Exception as e:
        print("[TEST] status error:", e)

    if not token or not dbid:
        print("[TEST] Thiếu token/dbid")
        return

    # Tạo payload động đúng schema thực tế của database
    q = "Ping từ script"
    a = "Nếu thấy page này là OK."
    try:
        payload = _build_dynamic_payload_force(dbid, q, a) 
    except Exception as e:
        print(f"[TEST] Build payload error:", e)
        return

    ok, code, body = _http_create_page(token, base, payload, timeout_s=15.0)
    print(f"[TEST] POST {base}/pages →", code, (body[:200] if isinstance(body, str) else body))



if __name__ == "__main__":
    print("🤖 Chatbot đã sẵn sàng! Gõ 'quit' để thoát.")
    conn = ensure_main_db()
    cur  = conn.cursor()
    build_book_embedding_index()
    build_major_embedding_index()

    #_test_push_notion_once()
    try:
        while True:
            sentence = input("Bạn: ").strip()
            if sentence.lower() == "quit":
                break
            print("Bot:", process_message(sentence))
    finally:
        conn.close()