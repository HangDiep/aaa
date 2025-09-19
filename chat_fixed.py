import random
import json
import torch
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words
import numpy as np
import sqlite3, datetime, os
from state_manager import StateManager
import requests  # NEW
# --------------------# ---- OLLAMA AUGMENT (append thêm câu trả lời) ----
USE_OLLAMA_AUGMENT = True           # bật/tắt tính năng bổ sung
OLLAMA_MODEL = "qwen2:1.5b"         # hoặc "llama3.2:3b"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

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

# khởi tạo giúp chat bot có cuộc trò chuyện

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
# Nói ngắn: đây là chỗ lưu “hộp thư đến” cho các câu hỏi mà chatbot chưa đồng bộ lên Notion.
def log_question_for_notation(question: str):
    """Ghi 1 câu hỏi vào 'inbox' để push lên Notion sau này (push_logs.py)."""
    if not question or not question.strip():
        return
    ensure_questions_log()
    conn2 = sqlite3.connect(FAQ_DB_PATH)
    cur2 = conn2.cursor()
    cur2.execute("INSERT INTO questions_log (question, synced) VALUES (?, 0)", (question.strip(),))
    conn2.commit(); conn2.close()
# 📌 Tóm lại: đây là hàm đưa câu hỏi vào hàng chờ (inbox). Sau này script push_logs.py sẽ đọc các bản ghi synced = 0 trong bảng này và đẩy lên Notion.
# --------------------
# MODEL
# --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding='utf-8-sig') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE, map_location=device)

input_size  = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words   = data["all_words"]
tags        = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# --------------------
# STATE / FLOW
# --------------------

def ollama_generate_append(base_reply: str, user_message: str) -> str:
    """
    Gọi Ollama để sinh thêm câu trả lời dựa trên base_reply và user_message.
    Trả về chuỗi bổ sung hoặc "" nếu lỗi/không có gì.
    """
    try:
        payload = {
            "model": OLLAMA_MODEL,
            # chỉ truyền trực tiếp 2 biến mà không kèm prompt hướng dẫn dài
            "prompt": f"Người dùng: {user_message}\nCâu trả lời hiện có: {base_reply}\n\nViết thêm:",
            "stream": False,
            "options": {
                "num_predict": 220
            }
        }
        r = requests.post(OLLAMA_URL, json=payload, timeout=25)
        if r.ok:
            txt = (r.json().get("response") or "").strip()
            if txt and txt.lower() not in (base_reply.lower()):
                return txt
    except Exception:
        pass
    return ""

try:
    state_mgr = StateManager("flows.json")
except Exception:
    state_mgr = StateManager()

INTERRUPT_INTENTS = set()
CANCEL_WORDS = {"hủy","huỷ","huy","cancel","thoát","dừng","đổi chủ đề","doi chu de"}
# ngắt hội tho
print("🤖 Chatbot đã sẵn sàng! Gõ 'quit' để thoát.")

try:
    while True:
        sentence = input("Bạn: ").strip()
        if sentence.lower() == "quit":
            break

        # Hủy flow thủ công
        if sentence.lower() in CANCEL_WORDS:
            try:
                state_mgr.exit_flow()
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

        reply = None
        tag_to_log = None
        confidence = 0.0

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
        if getattr(state_mgr, "active_flow", None):
            try:
                ctx_reply = state_mgr.handle(tag, sentence)
            except Exception:
                ctx_reply = None
            if ctx_reply:
                reply = ctx_reply
                tag_to_log = tag

        if reply is None and confidence > CONF_THRESHOLD:
            try:
                ctx_reply = state_mgr.handle(tag, sentence)
            except Exception:
                ctx_reply = None
            if ctx_reply:
                reply = ctx_reply
                tag_to_log = tag

        if reply is None:
            try:
                boot = state_mgr.bootstrap_by_text(sentence)
            except Exception:
                boot = None
            if boot:
                reply = boot

        if reply is None and confidence > CONF_THRESHOLD:
            resp_list = next((it["responses"] for it in intents["intents"] if it["tag"] == tag), None)
            if resp_list:
                reply = random.choice(resp_list)
                tag_to_log = tag

        if reply is None:
            reply = "Xin lỗi, mình chưa hiểu ý bạn."

        print("Bot:", reply)

        # LƯU LOG HỘI THOẠI (chat.db)
        cur.execute(
            "INSERT INTO conversations(user_message, bot_reply, intent_tag, confidence, time) VALUES (?,?,?,?,?)",
            (sentence, reply, tag_to_log, confidence, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()

        # GHI "INBOX CÂU HỎI" ĐỂ ĐẨY LÊN NOTION
        should_push_to_notion = (
            LOG_ALL_QUESTIONS or
            reply.strip().startswith("Xin lỗi, mình chưa hiểu") or
            confidence < CONF_THRESHOLD or
            tag_to_log is None
        )
        if should_push_to_notion:
            try:
                # ✅ gọi HÀM, không phải gán biến
                log_question_for_notation(f"User: {sentence}\nBot: {reply}")

            except Exception:
                pass

finally:
    conn.close()
