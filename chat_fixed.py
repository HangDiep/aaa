
import os
import random
import json
import sqlite3
import datetime
from typing import Optional

import numpy as np
import torch
import requests

from model import NeuralNet
from nltk_utils import tokenize, bag_of_words
from state_manager import StateManager
import requests  # NEW
# --------------------# ---- OLLAMA AUGMENT (append thêm câu trả lời) ----
USE_OLLAMA_AUGMENT = True           # bật/tắt tính năng bổ sung
OLLAMA_MODEL = "qwen2:1.5b"         # hoặc "llama3.2:3b"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# --------------------
# CẤU HÌNH LƯU LOG
# --------------------
CHAT_DB_PATH = "chat.db"

# Inbox câu hỏi để đẩy lên Notion
FAQ_DB_PATH = "D:/HTML/chat2/rag/faqs.db"   # giữ nguyên như push_logs.py

CONF_THRESHOLD = 0.60  # ngưỡng tự tin intent
LOG_ALL_QUESTIONS = True  # True = log mọi câu; False = chỉ log khi bot chưa hiểu / tự tin thấp

# --------------------
# DB: conversations (chat.db)
# --------------------
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

# --------------------
# DB: questions_log (faqs.db) - tạo nếu chưa có
# --------------------
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
    """Ghi 1 câu hỏi vào 'inbox' để push lên Notion sau này (push_logs.py)."""
    if not question or not question.strip():
        return
    ensure_questions_log()
    conn2 = sqlite3.connect(FAQ_DB_PATH)
    cur2 = conn2.cursor()
    cur2.execute("INSERT INTO questions_log (question, synced) VALUES (?, 0)", (question.strip(),))
    conn2.commit(); conn2.close()

# --------------------
# MODEL
# --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Đọc intents (ăn BOM nếu có)
with open("intents.json", "r", encoding="utf-8-sig") as f:
    intents = json.load(f)

# Load model đã train
FILE = "data.pth"
_data = torch.load(FILE, map_location=device)

input_size = _data["input_size"]
hidden_size = _data["hidden_size"]
output_size = _data["output_size"]
all_words = _data["all_words"]
tags = _data["tags"]
model_state = _data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# --------------------
# STATE / FLOW
# --------------------
try:
    state_mgr = StateManager("flows.json")
except Exception:
    state_mgr = StateManager()

INTERRUPT_INTENTS = set()
CANCEL_WORDS = {"hủy","huỷ","huy","cancel","thoát","dừng","đổi chủ đề","doi chu de"}

print("🤖 Chatbot đã sẵn sàng! Gõ 'quit' để thoát.")

conn = ensure_main_db()
cur = conn.cursor()

try:
    while True:
        sentence = input("Bạn: ").strip()
        lower_sentence = sentence.lower()

        if lower_sentence == "quit":
            break

        # Lệnh hủy luồng thủ công
        if lower_sentence in CANCEL_WORDS:
            try:
                state_mgr.exit_flow()
            except Exception:
                pass
            reply = "Đã hủy luồng hiện tại. Bạn muốn hỏi gì tiếp?"
            print("Bot:", reply)
            cur.execute(
                "INSERT INTO conversations(user_message, bot_reply, intent_tag, confidence, time) VALUES (?,?,?,?,?)",
                (sentence, reply, None, 0.0, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            conn.commit()
            continue

        # Khởi tạo
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

        # --- ƯU TIÊN NGỮ CẢNH ---
        # 0) Nếu đang ở trong flow: state manager xử lý TRƯỚC
        if getattr(state_mgr, "active_flow", None):
            try:
                ctx_reply = state_mgr.handle(tag, sentence)
            except Exception:
                ctx_reply = None
            if ctx_reply:
                reply = ctx_reply
                tag_to_log = tag

        # 1) Ưu tiên kiểm tra sách nếu chứa từ khóa
        if reply is None:
            book_keywords = [
                "sách", "tồn kho", "mượn",
                "cấu trúc dữ liệu", "trí tuệ nhân tạo", "lập trình python"
            ]
            if any(w in lower_sentence for w in book_keywords):
                inv = get_inventory_response(sentence)
                if not inv:  # thử theo từng keyword
                    for kw in sentence.split():
                        inv = get_inventory_response(kw)
                        if inv:
                            break
                if inv:
                    reply = inv
                    tag_to_log = "inventory_search"

        # 2) Tìm trong FAQ nếu câu hỏi thiên về thư viện
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

        # 3) Bootstrap theo từ khóa trong flows.json
        if reply is None:
            try:
                boot = state_mgr.bootstrap_by_text(sentence)
            except Exception:
                boot = None
            if boot:
                reply = boot

        # 4) Nếu vẫn chưa có -> dùng responses theo intent (chỉ khi đủ tự tin)
        if reply is None and confidence > CONF_THRESHOLD:
            resp_list = next((it["responses"] for it in intents["intents"] if it["tag"] == tag), None)
            if resp_list:
                reply = random.choice(resp_list)
                tag_to_log = tag

        # 5) Fallback cuối
        if reply is None:
            reply = "Xin lỗi, mình chưa hiểu ý bạn."

        print("Bot:", reply)

        # --- Lưu log ---
        cur.execute(
            "INSERT INTO conversations(user_message, bot_reply, intent_tag, confidence, time) VALUES (?,?,?,?,?)",
            (sentence, reply, tag_to_log, confidence, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()

        # --- Ghi inbox để đẩy lên Notion (đặt BÊN TRONG vòng lặp!) ---
        should_push_to_notion = (
            LOG_ALL_QUESTIONS
            or reply.strip().startswith("Xin lỗi, mình chưa hiểu")
            or confidence < CONF_THRESHOLD
            or tag_to_log is None
        )
        if should_push_to_notion:
            try:
                # ✅ gọi HÀM đúng tên (không gán biến)
                log_question_for_notion(f"User: {sentence}\nBot: {reply}")
            except Exception as e:
                # Không để logging làm hỏng flow chat
                print(f"[Notion inbox] Lỗi: {e}")

finally:
    try:
        conn.close()
    except Exception:
        pass
        