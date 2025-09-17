# chat_fixed.py (revised)
import random
import json
import torch
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words
import numpy as np
import sqlite3, datetime
from state_manager import StateManager

DB_PATH = "chat.db"
CONF_THRESHOLD = 0.60  # hạ tạm để dễ kích hoạt intent khi data còn mỏng

# --- Kết nối & chuẩn bị DB ---
conn = sqlite3.connect(DB_PATH)
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Đọc intents (ăn BOM nếu có)
with open('intents.json', 'r', encoding='utf-8-sig') as f:
    intents = json.load(f)

# Load model đã train
FILE = "data.pth"
data = torch.load(FILE, map_location=device)

input_size  = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words   = data["all_words"]
tags        = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)   # nạp trọng số
model.eval()

# State manager: cố gắng dùng flows.json nếu có
try:
    state_mgr = StateManager("flows.json")
except Exception:
    state_mgr = StateManager()

INTERRUPT_INTENTS = set()  # không ngắt flow bằng intent; chỉ hủy bằng CANCEL_WORDS
CANCEL_WORDS = {"hủy","huỷ","huy","cancel","thoát","dừng","đổi chủ đề","doi chu de"}

print("🤖 Chatbot đã sẵn sàng! Gõ 'quit' để thoát.")

try:
    while True:
        sentence = input("Bạn: ").strip()
        if sentence.lower() == "quit":
            break

        # Lệnh hủy luồng thủ công
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

        # Khởi tạo
        reply = None
        tag_to_log = None
        confidence = 0.0

        # --- NLU: dự đoán intent ---
        tokens = tokenize(sentence)
        X = bag_of_words(tokens, all_words)             # np.float32
        X = torch.from_numpy(X).unsqueeze(0).to(device) # (1, input_size)

        with torch.no_grad():
            output = model(X)                           # (1, num_classes)
            probs = torch.softmax(output, dim=1)
            prob, pred_idx = torch.max(probs, dim=1)
            tag = tags[pred_idx.item()]
            confidence = float(prob.item())

        # --- ƯU TIÊN NGỮ CẢNH ---
        # 0) Nếu đang ở trong flow: state manager xử lý TRƯỚC
        if getattr(state_mgr, "active_flow", None):
            # Không tự ý ngắt flow bằng intent; luôn cố gắng xử lý tiếp ngữ cảnh
            try:
                ctx_reply = state_mgr.handle(tag, sentence)
            except Exception:
                    ctx_reply = None
            if ctx_reply:
                    reply = ctx_reply
                    tag_to_log = tag

        # 1) Nếu chưa có reply & model tự tin: thử KHỞI ĐỘNG flow theo intent hiện tại
        if reply is None and confidence > CONF_THRESHOLD:
            try:
                ctx_reply = state_mgr.handle(tag, sentence)
            except Exception:
                ctx_reply = None
            if ctx_reply:
                reply = ctx_reply
                tag_to_log = tag

        # 2) Nếu vẫn chưa có reply: thử bootstrap theo từ khóa trong flows.json
        if reply is None:
            try:
                boot = state_mgr.bootstrap_by_text(sentence)
            except Exception:
                boot = None
            if boot:
                reply = boot
                # có thể chưa log intent vì chưa chắc chắn

        # 3) Nếu vẫn chưa có -> dùng responses theo intent (chỉ khi đủ tự tin)
        if reply is None and confidence > CONF_THRESHOLD:
            resp_list = next((it["responses"] for it in intents["intents"] if it["tag"] == tag), None)
            if resp_list:
                reply = random.choice(resp_list)
                tag_to_log = tag

        # 4) Fallback cuối
        if reply is None:
            reply = "Xin lỗi, mình chưa hiểu ý bạn."

        print("Bot:", reply)

        # --- Lưu log ---
        cur.execute(
            "INSERT INTO conversations(user_message, bot_reply, intent_tag, confidence, time) VALUES (?,?,?,?,?)",
            (sentence, reply, tag_to_log, confidence, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()

finally:
    conn.close()