import random
import json
import torch
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words
import numpy as np
import sqlite3, datetime
from state_manager import StateManager


# Nếu bạn dùng StateManager data-driven có flows.json:
# state_mgr = StateManager("flows.json")
# Nếu bạn dùng bản đơn giản không cần file ngoài:
state_mgr = StateManager("flows.json")  # hoặc StateManager() nếu không có flows.json

DB_PATH = "chat.db"

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
state_mgr = StateManager("flows.json")
model.eval()

print("🤖 Chatbot đã sẵn sàng! Gõ 'quit' để thoát.")

try:
    while True:
        sentence = input("Bạn: ")
        if sentence.strip().lower() == "quit":
            break

        tokens = tokenize(sentence)
        X = bag_of_words(tokens, all_words)                  # np.float32
        X = torch.from_numpy(X).unsqueeze(0).to(device)      # (1, input_size)

        with torch.no_grad():
            output = model(X)                                # (1, num_classes)
            probs = torch.softmax(output, dim=1)
            prob, pred_idx = torch.max(probs, dim=1)
            tag = tags[pred_idx.item()]
            confidence = float(prob.item())                  # đảm bảo là float thuần

        # --- Sinh câu trả lời ---
        reply = None

        if confidence > 0.60:
            # 1) ưu tiên ngữ cảnh (StateManager)
            ctx_reply = state_mgr.handle(tag, sentence)
            if ctx_reply is not None:# nếu state_mgr không xử lý → lấy response mặc định
                reply = ctx_reply
            else:
                # 2) không có context → dùng responses theo intent như cũ
                resp_list = next((it["responses"] for it in intents["intents"] if it["tag"] == tag), None)
                reply = random.choice(resp_list) if resp_list else "Xin lỗi, mình chưa hiểu ý bạn."
        else:
            # Fallback: không tự tin → coi như chưa hiểu
            reply = "Xin lỗi, mình chưa hiểu ý bạn."
            tag = None

        print("Bot:", reply)

        # --- Lưu log vào DB ---
        cur.execute(
            "INSERT INTO conversations(user_message, bot_reply, intent_tag, confidence, time) VALUES (?,?,?,?,?)",
            (sentence, reply, tag, confidence, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()

finally:
    conn.close()
