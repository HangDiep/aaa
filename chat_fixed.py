
import os, random, json, sqlite3, datetime
#chat_fixed.py
import numpy as np
import torch, requests
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words
from state_manager import StateManager
import threading
from dotenv import load_dotenv
from notion_client import Client
from typing import Optional, List, Dict
ENV_PATH = r"D:/HTML/chat2/rag/.env"
try:
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH)
except Exception:
    pass
_notion_cached = None

# Có thể đặt trong .env (ưu tiên .env) hoặc dùng default dưới đây
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2:1.5b")  # đổi thành model bạn đã pull
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "20"))  # giây
ENABLE_OLLAMA_APPEND = True  # bật/tắt việc cho Ollama viết thêm
MAX_OLLAMA_APPEND_TOKENS = 150  # số token tối đa Ollama được viết thêm
FAQ_API_URL = None
INVENTORY_API_URL = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#ng dùng hỏi bot trả lời lưu vào chat.db
CHAT_DB_PATH = os.path.join(BASE_DIR, "chat.db")
print(f"[ChatDB] Using: {CHAT_DB_PATH}")

DB_PATH = CHAT_DB_PATH  # dùng đúng đường dẫn DB
#Ghi các câu hỏi “chưa hiểu” hoặc “chờ duyệt”
FAQ_DB_PATH = os.path.normpath("D:/HTML/chat2/rag/faqs.db")
CONF_THRESHOLD = 0.60
LOG_ALL_QUESTIONS = True

FAQ_API_URL = "http://localhost:8000/search"
INVENTORY_API_URL = "http://localhost:8000/inventory"

INTERRUPT_INTENTS = set()
CANCEL_WORDS = {"hủy", "huỷ", "huy", "cancel", "thoát", "dừng", "đổi chủ đề", "doi chu de"}

# =========================
# DB helpers
# =========================
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


# =========================
# Model load
# =========================
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
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_faq_response(sentence: str) -> Optional[str]:
    """
    Gọi FAQ API và trả về kết quả dạng bảng text đẹp,
    thay vì JSON thô.
    """
    try:
        resp = requests.get(FAQ_API_URL, params={"q": sentence}, timeout=5)
        if resp.status_code != 200:
            print(f"[FAQ] HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        
        data = resp.json()
        if not isinstance(data, list) or not data:
            return None

        # Dựng bảng text
        lines: List[str] = []
        lines.append("📖 **Kết quả FAQ:**\n")
        lines.append("| Câu hỏi | Trả lời |")
        lines.append("|---------|---------|")

        for item in data:
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            if q or a:
                # Escape ký tự '|' để không phá bảng
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
# =========================
# CORE: xử lý 1 câu (web/CLI dùng chung)
# =========================
def process_message(sentence: str) -> str:
    sentence = (sentence or "").strip()
    if not sentence:
        return "Xin lỗi, mình chưa hiểu ý bạn."

    lower_sentence = sentence.lower()

    # KHỞI TẠO BIẾN TRƯỚC KHI DÙNG
    reply: Optional[str] = None
    tag_to_log: Optional[str] = None
    confidence: float = 0.0
    if reply is None or not str(reply).strip():
        reply = "Xin lỗi, mình chưa hiểu ý bạn."
    fallback_reply = "Xin lỗi, mình chưa hiểu ý bạn."
    if ENABLE_OLLAMA_APPEND and reply.strip() and reply.strip() != fallback_reply:
        base_reply = reply
        try:
            extra = ollama_generate_append(base_reply, sentence)
            if extra and extra.strip() and extra.strip() not in base_reply:
                reply = f"{base_reply.strip()} {extra.strip()}"
            else:
                reply = base_reply
        except Exception:
            reply = base_reply


    # Lưu log + push Notion (giữ nguyên như bạn đang làm)
    conn = ensure_main_db(); cur = conn.cursor()
    cur.execute(
        "INSERT INTO conversations(user_message, bot_reply, intent_tag, confidence, time) VALUES (?,?,?,?,?)",
        (sentence, reply, tag_to_log, confidence, _now()),
    )
    conn.commit(); conn.close()

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
def _get_notion_client():
    """
    Lazy-init Notion Client từ .env. Nếu thiếu token/DBID -> trả về None (không chặn luồng chat).
    """
    global _notion_cached
    if _notion_cached is not None:
        return _notion_cached

def _get_notion_client():
    """
    Lazy-init Notion Client từ .env. Nếu thiếu token/DBID -> trả về None (không chặn luồng chat).
    """
    global _notion_cached
    if _notion_cached is not None:
        return _notion_cached

    try:
        token = os.getenv("NOTION_TOKEN")
        dbid  = os.getenv("NOTION_DATABASE_ID")
        if token and dbid:
            _notion_cached = (Client(auth=token), dbid)
        else:
            print("⚠️ NOTION_TOKEN/NOTION_DATABASE_ID chưa có trong .env hoặc .env không tồn tại.")
            _notion_cached = None
    except Exception as e:
        print(f"⚠️ Lỗi khởi tạo Notion Client: {e}")
        _notion_cached = None
    return _notion_cached

def _rt(txt: str):
    return [{"type": "text", "text": {"content": txt or ""}}]

def push_to_notion(q: str, a: str):
    """
    Đẩy Q/A lên Notion. Không raise lỗi ra ngoài, để tránh làm hỏng luồng trả lời.
    """
    pair = _get_notion_client()
    if not pair:
        return
    client, dbid = pair
    q = (q or "").strip()
    a = (a or "").strip()
    if not q:
        return
    try:
        client.pages.create(
            parent={"database_id": dbid},
            properties={
                "Question": {"rich_text": _rt(q)},
                "Answer":   {"rich_text": _rt(a)},
                "Approved": {"checkbox": False},
                "Language": {"select": {"name": "Tiếng Việt"}},
            },
        )
        # dùng properties theo đúng schema DB của bạn
    except Exception as e:
        print(f"⚠️ Lỗi khi tạo page Notion: {e}")
def ollama_generate_append(base_reply: str, user_message: str) -> str:
    """
    Gọi Ollama để VIẾT THÊM 1–3 câu tiếng Việt, bám ngữ cảnh thư viện.
    Không thay thế nội dung chính; tránh bịa và KHÔNG mâu thuẫn dữ kiện có sẵn.
    Trả về chuỗi bổ sung hoặc "" nếu lỗi/không có gì.
    """
    if not ENABLE_OLLAMA_APPEND:
        return ""

    system_prompt = (
        "Bạn là trợ lý THƯ VIỆN Trường Đại học Tây Nguyên (DHTN).\n"
        "- Chỉ BỔ SUNG 1–2 câu, ngắn gọn, bám CÂU TRẢ LỜI GỐC.\n"
        "- Chỉ nói về: giờ mở/đóng, mượn–trả, thẻ thư viện, quy định, phí phạt, tra cứu, khu sách, liên hệ.\n"
        "- Nếu không chắc liên quan thư viện: TRẢ VỀ CHUỖI RỖNG.\n"
        "- KHÔNG bịa, KHÔNG quảng cáo, KHÔNG trả lời câu cá nhân/ngoài phạm vi.\n"
        "- Chỉ TIẾNG VIỆT. KHÔNG chuyển ngôn ngữ khác.\n"
        "- KHÔNG chào hỏi xã giao, KHÔNG dùng ngoặc kép, KHÔNG cảm thán."
    )


    # Dùng /api/generate của Ollama (đơn giản, latency thấp)
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    payload = {
    "model": OLLAMA_MODEL,
    "prompt": f"{system_prompt}\n\nNgười dùng: {user_message}\nCâu trả lời gốc:\n{base_reply}\n\nYêu cầu: Bổ sung 1–2 câu. Nếu không phù hợp, trả về trống.",
    "stream": False,
    "options": {
        "temperature": 0.1,          # bớt bay
        "top_p": 0.9,
        "repeat_penalty": 1.2,       # hạn chế lặp
        "num_predict": 80,           # ngắn gọn
        "stop": ["\n\n", "\"", "”", "“"]  # chặn xuống dòng dài, ngoặc kép
    }
}

    try:
        r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        if r.status_code != 200:
            print(f"[Ollama] HTTP {r.status_code}: {r.text[:200]}")
            return ""
        data = r.json()  # {"model": "...", "created_at": "...", "response": "...", ...}
        extra = (data.get("response") or "").strip()
        # Lọc bớt mô tả thừa
        if not extra:
            return ""
        # Chặn việc lặp lại y nguyên reply chính
        if extra in base_reply:
            return ""
        # Rút gọn 1–3 câu (phòng trường hợp model viết dài)
        # Tách theo dấu chấm. Nếu thấy xuống dòng, ghép lại.
        sentences = [s.strip() for s in extra.replace("\n", " ").split(".") if s.strip()]
        if not sentences:
            return ""
        extra_short = ". ".join(sentences[:3]).strip()
        if extra_short and not extra_short.endswith("."):
            extra_short += "."
        extra_short = sanitize_vi(extra_short)
        if not extra_short:
            return ""
        return extra_short
    except requests.RequestException as e:
        print(f"[Ollama] Lỗi kết nối: {e}")
        return ""
    except Exception as e:
        print(f"[Ollama] Lỗi xử lý: {e}")
        return ""
import re

def sanitize_vi(extra: str) -> str:
    if not extra: return ""
    # bỏ ký tự CJK/emoji
    extra = re.sub(r'[\u3400-\u9FFF\uF900-\uFAFF]+', '', extra)
    extra = re.sub(r'[\U0001F300-\U0001FAFF]', '', extra)
    # bỏ ngoặc kép + khoảng trắng thừa
    extra = extra.replace('“','').replace('”','').replace('"','').strip()
    extra = re.sub(r'\s+', ' ', extra)
    # bỏ câu chào/ xã giao
    banned_starts = ("chào mừng", "rất tiếc", "xin chào", "cảm ơn")
    if extra.lower().startswith(banned_starts): return ""
    # quá ngắn/ vô nghĩa
    if len(extra.split()) < 3: return ""
    return extra

# =========================
# CLI chỉ chạy khi gọi trực tiếp file
# =========================
if __name__ == "__main__":
    print("🤖 Chatbot đã sẵn sàng! Gõ 'quit' để thoát.")
    conn = ensure_main_db()
    cur  = conn.cursor()
    try:
        while True:
            sentence = input("Bạn: ").strip()
            if sentence.lower() == "quit":
                break
            print("Bot:", process_message(sentence))
    finally:
        conn.close()