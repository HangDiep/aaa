import random
import json
import torch
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words
import numpy as np
import sqlite3, datetime   # thêm để log vào DB

# --- Kết nối database ---
DB_PATH = "chat.db"
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_message TEXT,
    bot_reply   TEXT,
    intent_tag  TEXT,
    confidence  REAL,
    time TEXT
);
""")
conn.commit()
# ------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Đọc intents với UTF-8 (ăn cả BOM nếu có)
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

print("Chatbot đã sẵn sàng! Gõ 'quit' để thoát.")

while True:
    sentence = input("Bạn: ")
    if sentence.strip().lower() == "quit":
        break

    tokens = tokenize(sentence)
    X = bag_of_words(tokens, all_words)           # np.float32
    X = torch.from_numpy(X).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(X)
        probs = torch.softmax(output, dim=1)
        prob, pred_idx = torch.max(probs, dim=1)
        tag = tags[pred_idx.item()]
        confidence = prob.item()

    if confidence > 0.75:
        reply = None
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                reply = random.choice(intent["responses"])
                break
        if reply is None:
            reply = "Xin lỗi, mình chưa hiểu ý bạn."
            tag = None
    else:
        reply = "Xin lỗi, mình chưa hiểu ý bạn."
        tag = None

    print("Bot:", reply)

    # --- Lưu log vào DB ---
    cur.execute(
        "INSERT INTO conversations(user_message, bot_reply, intent_tag, confidence, time) VALUES (?,?,?,?,?)",
        (sentence, reply, tag, float(confidence), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()

# đóng kết nối khi thoát
conn.close()
