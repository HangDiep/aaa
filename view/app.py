# view/app.py
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
import sys
from fastapi.responses import HTMLResponse
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from chat_fixed import process_message


import os
import uuid

app = FastAPI()

# Mount static (giữ nguyên)
app.mount("/static", StaticFiles(directory="view"), name="static")

# Thêm đường dẫn project để import được chat_fixed.py và ocr_helper.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import não chat và OCR
from chat_fixed import process_message
from ocr_helper import ocr_from_image   # 🔹 THÊM DÒNG NÀY

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

# --------- ROUTE /chat: text + ảnh + OCR + gọi não chat_fixed ---------
@app.post("/chat")
async def chat(message: str = Form(""), image: UploadFile = File(None)):
    raw_text = (message or "").strip()

    # 1) Nếu có ảnh → lưu tạm + OCR
    image_path = None
    ocr_text = None
    if image and image.filename:
        suffix = Path(image.filename).suffix.lower()  # .jpg, .png, ...
        safe_filename = f"{uuid.uuid4()}{suffix}"
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        image_path = temp_dir / safe_filename

        content = await image.read()
        with open(image_path, "wb") as f:
            f.write(content)

        print(f"[UPLOAD] Đã lưu ảnh → {image_path}")

        # 🔹 OCR từ ảnh
        try:
            ocr_text = ocr_from_image(str(image_path))
        except Exception as e:
            print("[OCR] Lỗi khi quét ảnh:", e)
            ocr_text = None

    # 2) Ghép câu hỏi + OCR (nếu có)
    full_query = raw_text
    if ocr_text:
        if full_query:
            full_query += "\n\n[Thông tin đọc được từ ảnh]:\n" + ocr_text
        else:
            full_query = "[Thông tin đọc được từ ảnh]:\n" + ocr_text

    if not full_query:
        full_query = "Xin chào, mình chưa nhập gì cả."

    # 3) Gọi não chat_fixed (KHÔNG truyền image_path, đúng ý bạn)
    try:
        answer = process_message(full_query)
    finally:
        # 4) Xóa file ảnh tạm
        if image_path and image_path.exists():
            try:
                image_path.unlink()
                print(f"[CLEANUP] Đã xóa {image_path}")
            except:
                pass

    # 5) Trả về cho frontend (giữ cấu trúc đơn giản)
        # In thêm cho nhìn rõ trong terminal (tùy thích)
    print("[CHAT] User text:", raw_text)
    if ocr_text:
        print("[CHAT] OCR từ ảnh:", ocr_text)

        return {"answer": answer}

# ---- Các route cũ: giữ nguyên y chang ----
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
