# chat.py — MERGED (flows + FAQ/Inventory API + logs to chat.db & faqs.db)

import os
import random
import json
import sqlite3
import datetime
import requests
import torch
import numpy as np

from model import NeuralNet
from nltk_utils import tokenize, bag_of_words
from state_manager import StateManager

# ===================== CẤU HÌNH =====================
CHAT_DB_PATH   = "chat.db"                       # log hội thoại
FAQ_DB_PATH    = "D:/HTML/chat2/rag/faqs.db"     # inbox câu hỏi để đẩy Notion
INTENTS_PATH   = "intents.json"
WEIGHTS_PATH   = "data.pth"

CONF_THRESHOLD    = 0.60
LOG_ALL_QUESTIONS = True

CANCEL_WORDS = {"hủy","huỷ","huy","cancel","thoát","dừng","đổi chủ đề","doi chu de"}

# Từ khóa heuristics để nhận diện nhanh câu liên quan sách/FAQ
BOOK_KEYWORDS = [
    "sách", "tồn kho", "mượn", "trả", "đặt", "cấu trúc dữ liệu",
    "trí tuệ nhân tạo", "lập trình python"
]
FAQ_KEYWORDS = ["thư viện", "địa chỉ", "giờ", "liên hệ", "nội quy"]

# ===================== DB (chat.db) =====================
conn = sqlite3.connect(CHAT_DB_PATH)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_message TEXT,
    bot_reply   TEXT,
    intent_tag  TEXT,
    confidence  REAL,
    time        TEXT
);
""")
conn.commit()

# ===================== DB (faqs.db) =====================
def ensure_questions_log():
    os.makedirs(os.path.dirname(FAQ_DB_PATH), exist_ok=True)
    conn2 = sqlite3.connect(FAQ_DB_PATH)
    cur2 = conn2.cursor()
    cur2.execute("""
    CREATE TABLE IF NOT EXISTS questions_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question   TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        synced     INTEGER DEFAULT 0
    )
    """)
    conn2.commit(); conn2.close()

def log_question_for_notation(question: str):
    """Ghi 1 câu hỏi vào 'inbox' để push Notion sau (push_logs.py)."""
    if not question or not question.strip():
        return
    ensure_questions_log()
    conn2 = sqlite3.connect(FAQ_DB_PATH)
    cur2 = conn2.cursor()
    cur2.execute("INSERT INTO questions_log (question, synced) VALUES (?, 0)", (question.strip(),))
    conn2.commit(); conn2.close()

# ===================== MODEL / INTENTS =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(INTENTS_PATH, 'r', encoding='utf-8-sig') as f:
    intents = json.load(f)

data = torch.load(WEIGHTS_PATH, map_location=device)
input_size  = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words   = data["all_words"]
tags        = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# ===================== STATE / FLOW =====================
try:
    state_mgr = StateManager("flows.json")
except Exception:
    state_mgr = StateManager()

# ===================== API HELPERS =====================
def get_faq_response(sentence: str) -> str | None:
    """Gọi API FAQ (RAG) – trả về câu trả lời đầu tiên nếu có."""
    try:
        url = "http://localhost:8000/search"
        resp = requests.get(url, params={"q": sentence}, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                ans = data[0].get("answer")
                return ans if isinstance(ans, str) and ans.strip() else None
        return None
    except requests.RequestException:
        return None

def get_inventory_response(sentence: str) -> str | None:
    """Gọi API kiểm kho theo câu đầy đủ; nếu không ra có thể thử theo từ khóa."""
    try:
        url = "http://localhost:8000/inventory"
        resp = requests.get(url, params={"book_name": sentence}, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                b = data[0]
                # bảo vệ key
                name   = b.get("name")
                author = b.get("author", "?")
                year   = b.get("year", "?")
                qty    = b.get("quantity", "?")
                status = b.get("status", "?")
                if name:
                    return f"Sách '{name}' của tác giả {author}, năm xuất bản {year}, số lượng: {qty}, trạng thái: {status}"
        return None
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return None

print("🤖 Chatbot đã sẵn sàng! Gõ 'quit' để thoát.")

try:
    while True:
        sentence = input("Bạn: ").strip()
        if sentence.lower() == "quit":
            break

        # ===== HỦY FLOW THỦ CÔNG =====
        if sentence.lower() in CANCEL_WORDS:
            try:
                if hasattr(state_mgr, "exit_flow"):
                    state_mgr.exit_flow()
                elif hasattr(state_mgr, "_exit_flow"):
                    state_mgr._exit_flow()
                else:
                    setattr(state_mgr, "active_flow", None)
                    setattr(state_mgr, "current_state", None)
            except Exception:
                pass
            reply = "Đã hủy luồng hiện tại. Bạn muốn hỏi gì tiếp?"
            print("Bot:", reply)
            cur.execute(
                "INSERT INTO conversations(user_message, bot_reply, intent_tag, confidence, time) VALUES (?,?,?,?,?)",
                (sentence, reply, None, 0.0, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
            continue

        # ===== KHỞI TẠO =====
        reply = None
        tag_to_log = None
        confidence = 0.0

        # ===== NLU: dự đoán intent =====
        tokens = tokenize(sentence)
        X = bag_of_words(tokens, all_words)
        X = torch.from_numpy(X).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(X)
            probs = torch.softmax(output, dim=1)
            prob, pred_idx = torch.max(probs, dim=1)
            tag = tags[pred_idx.item()]
            confidence = float(prob.item())

        # ===== ƯU TIÊN NGỮ CẢNH (FLOW) =====
        # 0) Nếu đang ở flow: tiếp tục flow trước
        if getattr(state_mgr, "active_flow", None):
            try:
                ctx_reply = state_mgr.handle(tag, sentence)
            except Exception:
                ctx_reply = None
            if ctx_reply:
                reply = ctx_reply
                tag_to_log = tag

        # 1) Heuristic: nếu có vẻ hỏi về SÁCH → gọi INVENTORY API trước
        if reply is None and any(w in sentence.lower() for w in BOOK_KEYWORDS):
            inv = get_inventory_response(sentence)
            if not inv:
                # Thử tách từ khóa đơn lẻ
                for kw in sentence.split():
                    inv = get_inventory_response(kw)
                    if inv:
                        break
            if inv:
                reply = inv
                tag_to_log = "inventory_search"

        # 2) Heuristic: nếu có vẻ câu hỏi FAQ thư viện → gọi FAQ API
        if reply is None and any(w in sentence.lower() for w in FAQ_KEYWORDS):
            faq = get_faq_response(sentence)
            if not faq:
                for kw in sentence.split():
                    faq = get_faq_response(kw)
                    if faq:
                        break
            if faq:
                reply = faq
                tag_to_log = "faq_search"

        # 3) Nếu chưa có → thử khởi động/tiếp flow theo intent (khi tự tin)
        if reply is None and confidence > CONF_THRESHOLD:
            try:
                ctx_reply = state_mgr.handle(tag, sentence)
            except Exception:
                ctx_reply = None
            if ctx_reply:
                reply = ctx_reply
                tag_to_log = tag

        # 4) Nếu vẫn chưa có → bootstrap theo từ khóa trong flows.json
        if reply is None:
            try:
                boot = state_mgr.bootstrap_by_text(sentence)
            except Exception:
                boot = None
            if boot:
                reply = boot

        # 5) Nếu vẫn chưa có → dùng responses theo intent (khi đủ tự tin)
        if reply is None and confidence > CONF_THRESHOLD:
            resp_list = next((it["responses"] for it in intents["intents"] if it["tag"] == tag), None)
            if resp_list:
                reply = random.choice(resp_list)
                tag_to_log = tag

        # 6) Fallback cuối
        if reply is None:
            reply = "Xin lỗi, mình chưa hiểu ý bạn."

        print("Bot:", reply)

        # ===== LƯU LOG HỘI THOẠI (chat.db) =====
        cur.execute(
            "INSERT INTO conversations(user_message, bot_reply, intent_tag, confidence, time) VALUES (?,?,?,?,?)",
            (sentence, reply, tag_to_log, confidence, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()

        # ===== GHI 'INBOX CÂU HỎI' (faqs.db) để đẩy Notion sau =====
        should_push_to_notion = (
            LOG_ALL_QUESTIONS or
            reply.strip().startswith("Xin lỗi, mình chưa hiểu") or
            confidence < CONF_THRESHOLD or
            tag_to_log is None
        )
        if should_push_to_notion:
            try:
                log_question_for_notation(f"User: {sentence}\nBot: {reply}")
            except Exception:
                # không để lỗi logging làm hỏng phiên chat
                pass

finally:
    conn.close()
