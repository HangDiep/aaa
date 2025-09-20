import os, random, json, sqlite3, datetime
from typing import Optional

import numpy as np
import torch, requests

from model import NeuralNet
from nltk_utils import tokenize, bag_of_words
from state_manager import StateManager
import threading
from dotenv import load_dotenv
from notion_client import Client
# =========================
# Paths & Config
# =========================
FAQ_API_URL = None
INVENTORY_API_URL = None

ENV_PATH = r"D:/HTML/chat2/rag/.env"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_DB_PATH = os.path.join(BASE_DIR, "chat.db")
print(f"[ChatDB] Using: {CHAT_DB_PATH}")

DB_PATH = CHAT_DB_PATH  # dùng đúng đường dẫn DB
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

# =========================
# API helpers
# =========================
def get_faq_response(sentence: str) -> Optional[str]:
    try:
        resp = requests.get(FAQ_API_URL, params={"q": sentence}, timeout=5)
        if resp.status_code != 200:
            print(f"[FAQ] HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        if isinstance(data, list) and data:
            ans = data[0].get("answer")
            if ans:
                return ans
        return None
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

    # Hủy flow thủ công
    if lower_sentence in CANCEL_WORDS:
        try:
            state_mgr.exit_flow()
        except Exception:
            pass
        reply = "Đã hủy luồng hiện tại. Bạn muốn hỏi gì tiếp?"
        conn = ensure_main_db()
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO conversations(user_message, bot_reply, intent_tag, confidence, time) VALUES (?,?,?,?,?)",
            (sentence, reply, None, 0.0, _now()),
        )
        conn.commit(); conn.close()
        return reply

    reply: Optional[str] = None
    tag_to_log: Optional[str] = None
    confidence: float = 0.0

    # --- NLU: dự đoán intent ---
    tokens = tokenize(sentence)
    X = bag_of_words(tokens, all_words)
    X = torch.from_numpy(X).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(X)
        probs = torch.softmax(output, dim=1)
        prob, pred_idx = torch.max(probs, dim=1)
        tag = tags[pred_idx.item()]
        confidence = float(prob.item())

    # 0) Nếu đang có flow hoạt động → ưu tiên
    if getattr(state_mgr, "active_flow", None):
        try:
            ctx_reply = state_mgr.handle(tag, sentence)
        except Exception:
            ctx_reply = None
        if ctx_reply:
            reply = ctx_reply
            tag_to_log = tag

    # 1) Kiểm tra kho sách nếu có từ khóa
    if reply is None:
        book_keywords = ["sách", "tồn kho", "mượn", "cấu trúc dữ liệu", "trí tuệ nhân tạo", "lập trình python"]
        if any(w in lower_sentence for w in book_keywords):
            inv = get_inventory_response(sentence)
            if not inv:
                for kw in sentence.split():
                    inv = get_inventory_response(kw)
                    if inv:
                        break
            if inv:
                reply = inv
                tag_to_log = "inventory_search"

    # 2) FAQ thư viện
    if reply is None:
        faq_keywords = ["thư viện", "địa chỉ", "giờ", "liên hệ", "nội quy"]
        if any(w in lower_sentence for w in faq_keywords):
            faq = get_faq_response(sentence)
            if not faq:
                for kw in sentence.split():
                    faq = get_faq_response(kw)
                    if faq:
                        break
            if faq:
                reply = faq
                tag_to_log = "faq_search"

    # 3) Bootstrap theo flows.json
    if reply is None:
        try:
            boot = state_mgr.bootstrap_by_text(sentence)
        except Exception:
            boot = None
        if boot:
            reply = boot

    # 4) Trả lời theo intent khi tự tin đủ
    if reply is None and confidence > CONF_THRESHOLD:
        resp_list = next((it["responses"] for it in intents["intents"] if it["tag"] == tag), None)
        if resp_list:
            reply = random.choice(resp_list)
            tag_to_log = tag

    # 5) Fallback
    if reply is None:
        reply = "Xin lỗi, mình chưa hiểu ý bạn."

    # Lưu log hội thoại
    conn = ensure_main_db()
    cur  = conn.cursor()
    cur.execute(
        "INSERT INTO conversations(user_message, bot_reply, intent_tag, confidence, time) VALUES (?,?,?,?,?)",
        (sentence, reply, tag_to_log, confidence, _now()),
    )
    conn.commit(); conn.close()

    # Ghi inbox để đẩy lên Notion
    should_push = (
        LOG_ALL_QUESTIONS
        or reply.strip().startswith("Xin lỗi, mình chưa hiểu")
        or confidence < CONF_THRESHOLD
        or tag_to_log is None
    )
    if should_push:
        try:
            threading.Thread(target=push_to_notion, args=(sentence, reply), daemon=True).start()
        except Exception:
            pass

    return reply


_notion_cached = None
def _get_notion_client():
    """
    Lazy-init Notion Client từ .env. Nếu thiếu token/DBID -> trả về None (không chặn luồng chat).
    """
    global _notion_cached
    if _notion_cached is not None:
        return _notion_cached

    try:
        if os.path.exists(ENV_PATH):
            load_dotenv(ENV_PATH)
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
