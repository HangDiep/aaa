# view/app.py
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from pathlib import Path
import sys
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
app = FastAPI()

# Cho Python thấy thư mục cha: D:\HTML\1234 (nơi có chat_fixed.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from chat_fixed import process_message  # chat_fixed.py KHÔNG được import chính nó

app = FastAPI(title="Library Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ===== API =====
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(message: str = Form(...)):
    return {"answer": process_message(message)}  # front-end đọc data.answer

# Stub demo (có/không tùy bạn)
@app.get("/search")
def search(q: str):
    return [{"answer": "Giờ mở cửa: 7:30 - 17:00, Thứ 2–Thứ 6."}]

@app.get("/inventory")
def inventory(book_name: str):
    return [{"name": book_name, "author": "N/A", "year": "?", "quantity": 3, "status": "available"}]

# ===== STATIC =====
STATIC_DIR = Path(__file__).resolve().parent   # chính là thư mục view

# Serve các file tĩnh (nếu bạn có JS/CSS riêng). Không bắt route gốc.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 👉 Trả trực tiếp file Chatbot.html ở route "/"
@app.get("/", response_class=HTMLResponse)
def home():
    file_path = STATIC_DIR / "Chatbot.html"
    if not file_path.exists():
        return "<h1>❌ Không tìm thấy Chatbot.html</h1>"
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
    
@app.get("/ping")
def ping():
    return {"msg": "pong"}

