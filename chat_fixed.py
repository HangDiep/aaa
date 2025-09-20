
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


# =========================
# Config
# =========================
DB_PATH = "chat.db"
# 👉 Sửa đường dẫn này theo máy của bạn nếu cần
FAQ_DB_PATH = os.path.normpath("D:/HTML/chat2/rag/faqs.db")
CONF_THRESHOLD = 0.60  # tạm hạ để dễ kích hoạt intent khi data còn mỏng
LOG_ALL_QUESTIONS = True  # True = log mọi câu; False = chỉ log khi bot chưa hiểu / tự tin thấp

# API endpoints (FastAPI backend)
FAQ_API_URL = "http://localhost:8000/search"
INVENTORY_API_URL = "http://localhost:8000/inventory"

# Flow control
INTERRUPT_INTENTS = set()  # không ngắt flow bằng intent; chỉ hủy bằng CANCEL_WORDS
CANCEL_WORDS = {"hủy", "huỷ", "huy", "cancel", "thoát", "dừng", "đổi chủ đề", "doi chu de"}


# =========================
# DB helpers
# =========================
def ensure_main_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
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
    """Ghi 1 câu hỏi + trả lời vào inbox để push lên Notion (push_logs.py)."""
    if not question or not question.strip():
        return
    ensure_questions_log_db()
    try:
        conn2 = sqlite3.connect(FAQ_DB_PATH)
        cur2 = conn2.cursor()
        cur2.execute(
            "INSERT INTO questions_log (question, synced) VALUES (?, 0)",
            (question.strip(),),
        )
        conn2.commit()
    finally:
        try:
            conn2.close()
        except Exception:
            pass


# =========================
# Model load
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# State manager: cố gắng dùng flows.json nếu có
try:
    state_mgr = StateManager("flows.json")
except Exception:
    state_mgr = StateManager()


# =========================
# API helpers
# =========================
def get_faq_response(sentence: str) -> Optional[str]:
    try:
        resp = requests.get(FAQ_API_URL, params={"q": sentence}, timeout=5)
        if resp.status_code == 200:
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
        if resp.status_code == 200:
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
# Runtime
# =========================
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
        