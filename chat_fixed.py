import os, random, json, sqlite3, re, time
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

def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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



# def classify_category(user_message: str) -> str:
#     url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
#     msg = user_message.lower().strip()

#     categories = get_all_categories()
#     # ghép thành bullet list
#     bullet = "\n".join(f"- {c}" for c in categories)

#     system_prompt = (
#         "Bạn là trợ lý ảo của Thư viện.\n"
#         "Nhiệm vụ: Đọc câu hỏi của người dùng và CHỈ TRẢ VỀ TÊN MỘT Category "
#         "trong danh sách sau (phải chọn đúng 1):\n\n"
#         f"{bullet}\n\n"
#         "- \"Thông tin ngành\": câu hỏi về ngành đào tạo, tên ngành, mã ngành, mô tả ngành, "
#         "số ngành, ngành nào có đào tạo...\n"
#         "- \"Tra cứu sách\": câu hỏi về tên sách, danh sách sách, sách thuộc ngành nào, "
#         "sách còn hay hết...\n"
#         "Nếu câu hỏi chỉ là tên một ngành (ví dụ: \"Công nghệ thông tin\", \"Y học\"), "
#         "thì chọn \"Thông tin ngành\".\n"
#         "Nếu không chắc chắn -> chọn Category có vẻ gần nhất. "
#         "Nếu hoàn toàn không phù hợp -> chọn 'Chưa phân loại' nếu có.\n\n"
#         "Hãy TRẢ VỀ duy nhất tên Category, không giải thích thêm."
#     )


#     user_prompt = f"Câu hỏi người dùng: \"{user_message}\""

#     payload = {
#         "model": OLLAMA_MODEL,
#         "prompt": system_prompt + "\n\n" + user_prompt,
#         "stream": False,
#         "options": {
#             "temperature": 0.0,
#             "num_predict": 32
#         }
#     }

#     try:
#         r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
#         resp = (r.json().get("response") or "").strip()
#         cat = resp.splitlines()[0].strip()
#         return cat or "Chưa phân loại"
#     except Exception as e:
#         print("[Category] Error:", e)
#         return "Chưa phân loại"
    
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


# def classify_category(user_message: str) -> str:
#     """
#     Phân loại ý định (Intent) dựa trên ý nghĩa, không dùng keyword matching.
#     Dùng LLM để suy luận ngữ nghĩa – hiểu cả sai chính tả, viết tắt, câu hỏi thiếu.
#     """

#     url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    
#     categories = get_all_categories()
#     if not categories:
#         return "Chưa phân loại"

#     bullet = "\n".join(f"- {c}" for c in categories)

#     system_prompt = f"""
# Bạn là trợ lý ảo của Thư viện. Bạn phân tích Ý NGHĨA câu hỏi, không dựa trên từ khóa.
# Nhiệm vụ: chọn CHÍNH XÁC 1 category phù hợp nhất từ danh sách sau:

# {bullet}

# === HƯỚNG DẪN HIỂU Ý (KHÔNG DÙNG KEYWORD) ===
# - "Thông tin ngành": khi người dùng nhắc tên ngành (CNTT, Công nghệ thông tin, Y học…)
#   kể cả họ chỉ viết tên ngành CHỮNG KHÔNG có từ “ngành”.
#   Hiểu cả sai chính tả: "công nghê thông tin", "C.N.T.T", "cnt"
#   Các câu dạng:
#     • "CNTT là gì?"
#     • "Ngành đó có những môn nào?"
#     • "Kinh tế học ra làm gì?"
#     • "khối y có bao nhiêu ngành?"
# - "Tra cứu sách": khi người dùng hỏi về danh sách sách, tên sách, sách của ngành nào,
#   sách còn không, sách tác giả nào…
#   Hiểu cả câu không rõ:
#     • "Có sách AI không?"
#     • "liệt kê sách CNTT"
#     • "Python còn không?"
#     • "mạng máy tính có bao nhiêu bản?"
# - Nếu user chỉ nhập 1 từ hoặc cụm từ:
#   → Nếu giống một ngành → chọn "Thông tin ngành"
#   → Nếu giống tên sách → chọn "Tra cứu sách"

# - Nếu câu hỏi không thuộc 2 nhóm trên, chọn category có vẻ phù hợp nhất
# - Nếu hoàn toàn không chắc → trả về: "Chưa phân loại"

# CHỈ TRẢ VỀ DUY NHẤT TÊN CATEGORY.
# Không giải thích thêm.
# """

#     user_prompt = f"Câu hỏi người dùng: \"{user_message}\""

#     payload = {
#         "model": OLLAMA_MODEL,
#         "prompt": system_prompt + "\n" + user_prompt,
#         "stream": False,
#         "options": {"temperature": 0.0, "num_predict": 32},
#     }

#     try:
#         r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
#         raw = (r.json().get("response") or "").strip()
#         cat = raw.splitlines()[0].strip()
#         return cat or "Chưa phân loại"
#     except:
#         return "Chưa phân loại"
def classify_category(user_message: str) -> str:
    """
    Phân loại intent 100% theo NGỮ NGHĨA bằng LLM.
    Không dùng keyword.
    Hiểu cả khi user chỉ gõ tên ngành hoặc tên sách.
    """
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    msg = user_message.strip()

    # Lấy danh sách ngành từ DB → tự học
    conn = sqlite3.connect(FAQ_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name FROM majors")
    major_names = [r[0] for r in cur.fetchall()]
    conn.close()

    bullet = "\n".join(f"- {m}" for m in major_names)

    system_prompt = f"""
Bạn là trợ lý thư viện.
Nhiệm vụ: Phân tích câu của người dùng và chọn chính xác 1 trong 2 category sau:

1) "Thông tin ngành"  → khi người dùng:
    - Gõ tên ngành (vd: Công nghệ thông tin, CNTT, Y học…)
    - Viết sai chính tả nhưng gần giống tên ngành
    - Hỏi mô tả ngành, ngành học gì, ra làm gì,…

2) "Tra cứu sách" → khi người dùng:
    - Hỏi về sách, tên sách, liệt kê sách
    - Hỏi sách còn không, sách ngành nào
    - Hỏi sách theo tác giả, năm xuất bản,…

=== Danh sách ngành hợp lệ ===
{bullet}

QUY TẮC:
- Nếu user chỉ gõ 1 từ/cụm từ và giống với **tên ngành** → chọn "Thông tin ngành".
- Nếu user hỏi loại sách / tên sách → chọn "Tra cứu sách".
- Trả về duy nhất 1 trong 2 chuỗi:
    Thông tin ngành
    Tra cứu sách
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": system_prompt + "\n\nCâu của user: " + msg,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 32}
    }

    try:
        r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        raw = (r.json().get("response") or "").strip().splitlines()[0]
        return raw if raw in ["Thông tin ngành", "Tra cứu sách"] else "Tra cứu sách"
    except:
        return "Tra cứu sách"



def answer_from_books(user_message: str) -> str:
    try:
        text = user_message.strip().lower()
        conn = sqlite3.connect(FAQ_DB_PATH)
        cur = conn.cursor()

        # ========== 1) Kiểm tra user nhập tên NGÀNH ==========
        cur.execute("SELECT name, major_id FROM majors")
        majors = cur.fetchall()

        for name, major_id in majors:
            if text == name.lower().strip():
                cur.execute("""
                    SELECT name, author, year, quantity, status
                    FROM books
                    WHERE major_id = ?
                """, (major_id,))
                books = cur.fetchall()
                conn.close()

                if not books:
                    return f"Ngành {name} hiện chưa có sách."

                block = "\n".join(
                    f"- {n} – {a}, {y}, SL: {q}, Trạng thái: {s}"
                    for n, a, y, q, s in books
                )
                return f"Danh sách sách thuộc ngành **{name}**:\n\n{block}"

        # ========== 2) Kiểm tra user hỏi TÁC GIẢ ==========
        cur.execute("SELECT DISTINCT author FROM books")
        authors = [r[0].lower().strip() for r in cur.fetchall()]

        for a in authors:
            if a in text or text in a:
                cur.execute("""
                    SELECT name, year, quantity, status
                    FROM books
                    WHERE lower(author) = ?
                """, (a,))
                books = cur.fetchall()
                conn.close()

                if not books:
                    return f"Tác giả {a} chưa có sách trong hệ thống."

                block = "\n".join(
                    f"- {n}, {y}, SL: {q}, Trạng thái: {s}"
                    for n, y, q, s in books
                )
                return f"Các sách của tác giả **{a}**:\n\n{block}"

        # ========== 3) Kiểm tra user nhập TÊN SÁCH ==========
        cur.execute("""
            SELECT name, author, year, quantity, status, major_id
            FROM books
        """)
        rows = cur.fetchall()

        for n, a, y, q, s, m in rows:
            if text == n.lower().strip():
                # Tìm tên ngành
                cur.execute("SELECT name FROM majors WHERE major_id = ?", (m,))
                major_name = cur.fetchone()
                major_label = major_name[0] if major_name else "Không rõ"

                conn.close()
                return (
                    f"**Thông tin sách:**\n"
                    f"- Tên: {n}\n"
                    f"- Tác giả: {a}\n"
                    f"- Năm XB: {y}\n"
                    f"- Số lượng: {q}\n"
                    f"- Trạng thái: {s}\n"
                    f"- Ngành: {major_label}"
                )

        # ========== 4) Nếu không trùng gì → dùng LLM trích keyword ==========
        extract_prompt = f"""
Bạn là trợ lý thư viện.
Nhiệm vụ: trích TÊN SÁCH hoặc TÊN TÁC GIẢ từ câu hỏi sau.

Chỉ trả về:
- Hoặc 1 tên sách
- Hoặc 1 tên tác giả
- Nếu không tìm thấy → trả rỗng

Câu hỏi: "{user_message}"
"""

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": extract_prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 50}
        }
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT)
        key = (r.json().get("response") or "").split("\n")[0].strip()
        key = key.lower()

        if not key:
            return "Không tìm thấy thông tin phù hợp."

        # ========== 5) Tìm theo tên sách gần đúng ==========
        cur = sqlite3.connect(FAQ_DB_PATH).cursor()
        cur.execute("""
            SELECT name, author, year, quantity, status, major_id
            FROM books
            WHERE lower(name) LIKE ?
        """, (f"%{key}%",))
        book_rows = cur.fetchall()

        if book_rows:
            n, a, y, q, s, m = book_rows[0]
            cur.execute("SELECT name FROM majors WHERE major_id = ?", (m,))
            major = cur.fetchone()
            major_label = major[0] if major else "Không rõ"

            return (
                f"**Thông tin sách:**\n"
                f"- Tên: {n}\n"
                f"- Tác giả: {a}\n"
                f"- Năm XB: {y}\n"
                f"- Số lượng: {q}\n"
                f"- Trạng thái: {s}\n"
                f"- Ngành: {major_label}"
            )

        # Không tìm thấy gì cả
        return f"Không tìm thấy sách liên quan tới: **{key}**"

    except Exception as e:
        return f"[LỖI books] {e}"



def process_message(sentence: str) -> str:
    sentence = (sentence or "").strip()
    if not sentence:
        return "Xin lỗi, mình chưa hiểu ý bạn."

    reply: Optional[str] = None
    tag_to_log: Optional[str] = None
    confidence: float = 0.0

    if ollama_alive():
        try:
            # 1) Phân loại Category (Intent) bằng LLM
            category = classify_category(sentence)
            tag_to_log = category  # luôn log intent = category

            # 2) Rẽ nhánh theo category lớn
            
            if category == "Thông tin ngành":
                major_reply = answer_from_majors(sentence)
                if major_reply:
                    reply = major_reply
                    tag_to_log = category

            elif category == "Tra cứu sách":
                book_reply = answer_from_books(sentence)
                if book_reply:
                    reply = book_reply
                    tag_to_log = category
            else:
                # 3) Mặc định: dùng FAQ (Notion → faq.db)
                conn_faq = sqlite3.connect(FAQ_DB_PATH)   # faq.db
                cur_faq = conn_faq.cursor()
                cur_faq.execute("""
                    SELECT question, answer
                    FROM faq
                    WHERE category = ?
                      AND (approved = 1 OR approved IS NULL)
                """, (category,))
                rows = cur_faq.fetchall()
                conn_faq.close()

                if rows:
                    # Ghép block Q/A cho LLM đọc và trả lời
                    faq_block_lines = []
                    for idx, (q, a) in enumerate(rows, start=1):
                        q = (q or "").strip()
                        a = (a or "").strip()
                        faq_block_lines.append(f"{idx}) Q: {q}\n   A: {a}")
                    faq_block = "\n\n".join(faq_block_lines)

                    answer_prompt = (
                        "Bạn là chatbot của Thư viện.\n"
                        f"CÂU HỎI CỦA NGƯỜI DÙNG:\n{sentence}\n\n"
                        f"DANH SÁCH CÂU HỎI – CÂU TRẢ LỜI TRONG CATEGORY \"{category}\":\n\n"
                        f"{faq_block}\n\n"
                        "NHIỆM VỤ:\n"
                        "1. Đọc kỹ câu hỏi của người dùng và các Answer (A) ở trên.\n"
                        "2. Trả lời dựa trên NỘI DUNG các Answer này. Có thể ghép thông tin từ nhiều Answer nếu cần.\n"
                        "3. KHÔNG được bịa thêm thông tin ngoài những gì có trong Answer.\n"
                        "4. Nếu không có Answer nào phù hợp, hãy nói: "
                        "\"Hiện tại mình chưa có thông tin chính xác trong hệ thống thư viện về câu hỏi này.\"\n\n"
                        "Bây giờ hãy trả lời người dùng:"
                    )

                    payload = {
                        "model": OLLAMA_MODEL,
                        "prompt": answer_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.2,
                            "num_predict": 200
                        }
                    }

                    r = requests.post(f"{OLLAMA_URL.rstrip('/')}/api/generate",
                                      json=payload, timeout=OLLAMA_TIMEOUT)
                    if r.status_code == 200:
                        reply_llm = (r.json().get("response") or "").strip()
                        if reply_llm:
                            reply = reply_llm
                            confidence = 0.9  # tạm gán cao, sau chỉnh sau
        except Exception as e:
            print("[process_message] FAQ/LLM error:", e)

    # 2) Fallback: nếu vẫn chưa có câu trả lời, trả mặc định
    if reply is None or not reply.strip():
        reply = "Xin lỗi, mình chưa hiểu ý bạn."
        confidence = 0.0

    # 3) Append thêm câu cho mượt, dùng hàm cũ (nếu bật)
    if ENABLE_OLLAMA_APPEND and reply.strip() and ollama_alive():
        extra = ollama_generate_continuation(reply, sentence, max_sentences=3)
        if extra:
            reply = f"{reply.strip()} {extra.strip()}"
        


    # 4) Ghi SQLite (bảng conversations) như trước
    conn = ensure_main_db()
    cur  = conn.cursor()
    cur.execute(
        "INSERT INTO conversations(user_message, bot_reply, intent_tag, confidence, time) "
        "VALUES (?,?,?,?,?)",
        (sentence, reply, tag_to_log, confidence, _now()),
    )
    conn.commit()
    conn.close()

    # 5) Ghi thêm vào faq.db (inbox) như cũ
    try:
        log_question_for_notion(f"User: {sentence}\nBot: {reply}")
    except Exception as e:
        print(f"[Notion inbox] Lỗi ghi faq.db: {e}")

    # 6) Đẩy Notion (không chặn luồng chat) – giữ logic cũ
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
    #_test_push_notion_once()
    try:
        while True:
            sentence = input("Bạn: ").strip()
            if sentence.lower() == "quit":
                break
            print("Bot:", process_message(sentence))
    finally:
        conn.close()