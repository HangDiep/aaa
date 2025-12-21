# ==========================================
# HO TÊN: Đỗ Thị Hồng Điệp
# MSSV: 23103014
# ĐỒ ÁN: Chatbot Dynamic Router - TTN University
# NGÀY NỘP: 21/12/2025
# Copyright © 2025. All rights reserved.
# ==========================================

# view/app.py
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
import sys
from fastapi.responses import HTMLResponse
# from chat_fixed import process_message (Đã bỏ để import ở dưới sau khi append path)

import os
import uuid
app = FastAPI()

# Mount static
app.mount("/static", StaticFiles(directory="view"), name="static")

# Thêm đường dẫn project để import được chat_fixed.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from chat_fixed import process_message

# Bây giờ mới import, vì process_message đã được cập nhật nhận image_path

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= BACKGROUND SYNC (3-MIN AUTO SCAN) =================
import asyncio
from sync_dynamic import scan_new_databases

@app.on_event("startup")
async def startup_event():
    """Khởi động quét Notion tự động mỗi 3 phút."""
    asyncio.create_task(run_auto_scan_loop())

async def run_auto_scan_loop():
    # Lấy interval từ .env hoặc mặc định 180s (3 phút)
    interval = int(os.getenv("SYNC_INTERVAL_SECONDS", 180))
    print(f"⏰ [View Server] Auto-Scan started (Every {interval}s)")
    
    while True:
        try:
            print(f"\n⏰ [Auto-Scan] Triggering scheduled scan...")
            await scan_new_databases()
        except Exception as e:
            print(f"❌ [Auto-Scan] Error in view server loop: {e}")
        
        await asyncio.sleep(interval)
# =====================================================================

@app.get("/health")
def health():
    return {"status": "ok"}

# Route mới – nhận cả text và ảnh
@app.post("/chat")
async def chat(message: str = Form(""), image: UploadFile = File(None)):
    image_path = None
    if image and image.filename:
        # Đổi tên thành UUID + đuôi gốc → 100% không lỗi Unicode
        suffix = Path(image.filename).suffix.lower()  # .jpg, .png, ...
        safe_filename = f"{uuid.uuid4()}{suffix}"
        os.makedirs("temp", exist_ok=True)
        image_path = Path("temp") / safe_filename
        
        content = await image.read()
        with open(image_path, "wb") as f:
            f.write(content)
        
        print(f"[UPLOAD] Đã lưu ảnh → {image_path}")

    try:
        answer = process_message(message.strip(), image_path=str(image_path) if image_path else None)
    finally:
        # Luôn xóa file tạm sau khi dùng xong (tránh đầy ổ)
        if image_path and image_path.exists():
            try:
                image_path.unlink()
                print(f"[CLEANUP] Đã xóa {image_path}")
            except:
                pass

    return {"answer": answer}
@app.get("/", response_class=HTMLResponse)
def home():
    file_path = Path(__file__).parent / "index.html"
    if not file_path.exists():
        return "<h1>Không tìm thấy index.html</h1>"
    return file_path.read_text(encoding="utf-8")
# Các route cũ
@app.get("/search")
def search(q: str):
    return [{"answer": "Giờ mở cửa: 7:30 - 17:00, Thứ 2–Thứ 6."}]

@app.get("/inventory")
def inventory(book_name: str):
    return [{"name": book_name, "author": "N/A", "year": "?", "quantity": 3, "status": "available"}]

# Serve HTML
STATIC_DIR = Path(__file__).resolve().parent
@app.get("/chatbot", response_class=HTMLResponse)
def chatbot_page():
    file_path = Path(__file__).parent / "Chatbot.html"
    if not file_path.exists():
        return "<h1>Không tìm thấy Chatbot.html</h1>"
    return file_path.read_text(encoding="utf-8")
@app.get("/chatbot", response_class=HTMLResponse)
def chatbot_page():
    file_path = Path(__file__).parent / "Chatbot.html"
    return file_path.read_text(encoding="utf-8")

@app.get("/ping")
def ping():
    return {"msg": "pong"}