# view/app.py
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
import sys
import os
import uuid
app = FastAPI()

# Mount static
app.mount("/static", StaticFiles(directory="view"), name="static")

# Thêm đường dẫn project để import được chat_fixed.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Bây giờ mới import, vì process_message đã được cập nhật nhận image_path
from chat_fixed import process_message

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Các route cũ
@app.get("/search")
def search(q: str):
    return [{"answer": "Giờ mở cửa: 7:30 - 17:00, Thứ 2–Thứ 6."}]

@app.get("/inventory")
def inventory(book_name: str):
    return [{"name": book_name, "author": "N/A", "year": "?", "quantity": 3, "status": "available"}]

# Serve HTML
STATIC_DIR = Path(__file__).resolve().parent
@app.get("/", response_class=HTMLResponse)
def home():
    file_path = STATIC_DIR / "Chatbot.html"
    if not file_path.exists():
        return "<h1>Không tìm thấy Chatbot.html</h1>"
    return file_path.read_text(encoding="utf-8")

@app.get("/ping")
def ping():
    return {"msg": "pong"}