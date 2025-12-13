# ======================
#  LIBRARY CHATBOT API (FULL CHAT + OCR + WEBSOCKET VOICE)
# ======================

import sys
import os
import io
import json
import base64
import time
import numpy as np
from pathlib import Path
from fastapi import FastAPI, Form, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# -----------------------
#  SETUP PATHS & IMPORTS
# -----------------------
# Cho Python thấy thư mục cha: D:\HTML\a_Copy (nơi có chat_fixed.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import logic từ chat_fixed (hoặc chat.py)
try:
    from chat_fixed import process_message
except ImportError:
    # Fallback nếu không import được chat_fixed
    from chat import process_message

# -----------------------
#  MODELS (Whisper, EasyOCR)
# -----------------------
# OCR calls helper
try:
    from ocr_helper import ocr_from_image
except ImportError:
    ocr_from_image = None


app = FastAPI(title="Library Chat API (View Layer)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -----------------------
#  STARTUP EVENT (AUTO-SYNC)
# -----------------------
#đoạn kích hoạt tự động cập nhật 
import asyncio
@app.on_event("startup")
async def startup_event():
    try:
        from chat_fixed import run_auto_scan_loop
        asyncio.create_task(run_auto_scan_loop())
        print("✅ Auto-scan loop started from view/app.py")
    except ImportError:
        print("⚠️ Could not import run_auto_scan_loop from chat_fixed")

# -----------------------
#  API: CHAT
# -----------------------
@app.post("/chat")
def chat_endpoint(message: str = Form(...), session_id: str = Form("default")):
    # Proxy tới logic xử lý tin nhắn
    return {"answer": process_message(message, session_id=session_id)}

# -----------------------
#  API: OCR
# -----------------------
@app.post("/ocr")
async def ocr_endpoint(image: UploadFile = File(...)):
    if not ocr_from_image:
         return {"answer": "OCR helper not found."}
    
    try:
        # Lưu temporary
        temp_filename = f"temp_view_ocr_{int(time.time())}_{image.filename}"
        with open(temp_filename, "wb") as f:
            f.write(await image.read())
            
        text = ocr_from_image(temp_filename)
        
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        if not text:
            return {"answer": "Không đọc được chữ nào."}
            
        # Hỏi chatbot
        reply = process_message(f"Nội dung trong ảnh là: {text}. Hãy phân tích và trả lời.", session_id="default")
        return {"answer": reply}
    except Exception as e:
        return {"answer": f"Lỗi OCR: {e}"}

# -----------------------
#  WEBSOCKET: VOICE (Mounted from banghiamcuoicung)
# -----------------------
try:
    from banghiamcuoicung.server import router as voice_router
    app.include_router(voice_router)
    print("✅ Voice router mounted from banghiamcuoicung")
except ImportError as e:
    print(f"⚠️ Failed to mount voice router: {e}") 
    # Fallback dummy if needed, but better to fail explicitly or warn

# -----------------------
#  DUMMY ENDPOINTS
# -----------------------
@app.get("/search")
def search(q: str):
    return [{"answer": "Giờ mở cửa: 7:30 - 17:00, Thứ 2–Thứ 6."}]

@app.get("/inventory")
def inventory(book_name: str):
    return [{"name": book_name, "author": "N/A", "year": "?", "quantity": 3, "status": "available"}]

@app.get("/ping")
def ping():
    return {"msg": "pong"}

# -----------------------
#  STATIC & HOME
# -----------------------
STATIC_DIR = Path(__file__).resolve().parent
app.mount("/view", StaticFiles(directory=str(STATIC_DIR)), name="view")

@app.get("/", response_class=HTMLResponse)
def home():
    file_path = STATIC_DIR / "Chatbot.html"
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>❌ Không tìm thấy Chatbot.html</h1>"