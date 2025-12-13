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
# Tương tự chat_fixed, ta load model ở đây hoặc import từ shared helper
# Để tránh duplicate code và memory, tốt nhất nên gọi endpoint của chat_fixed nếu chạy server kia.
# Nhưng nếu user muốn chạy file này độc lập, ta cần load lại.
try:
    from faster_whisper import WhisperModel
    from pydub import AudioSegment
    whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    print("🔊 Whisper loaded in view/app.py")
except Exception as e:
    print(f"⚠️ Whisper load failed: {e}")
    whisper_model = None

# OCR sẽ gọi qua helper để tiết kiệm RAM hoặc dùng endpoint /ocr của chat_fixed
# Ở đây ta implement gọi trực tiếp helper cho tiện
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
#  WEBSOCKET: VOICE
# -----------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🟢 WS connected (view/app.py)")
    
    buffer = b""
    last_time = None
    SILENCE_GAP = 0.55
    MIN_SIZE = 4000
    MAX_SIZE = 150000

    try:
        while True:
            data = await websocket.receive_text()
            chunk = base64.b64decode(data)
            buffer += chunk
            
            now = time.time()
            if last_time is None: last_time = now
            
            if now - last_time > SILENCE_GAP and len(buffer) > MIN_SIZE:
                if whisper_model:
                    try:
                        audio = AudioSegment.from_file(io.BytesIO(buffer), format="webm")
                        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                        samples /= 32768.0
                        
                        segments, _ = whisper_model.transcribe(samples, language="vi", vad_filter=True)
                        text = "".join(seg.text for seg in segments).strip()
                        
                        if text:
                            print(f"🎤 User (Voice): {text}")
                            await websocket.send_text(json.dumps({"sender": "user", "text": text}, ensure_ascii=False))
                            
                            answer = process_message(text, session_id="voice_session")
                            await websocket.send_text(json.dumps({"sender": "bot", "text": answer}, ensure_ascii=False))
                    except Exception as e:
                        print("❌ Voice error:", e)
                
                buffer = b""
            
            if len(buffer) > MAX_SIZE: buffer = b""
            last_time = now
            
    except Exception:
        print("🔴 WS closed")

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
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    file_path = STATIC_DIR / "Chatbot.html"
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>❌ Không tìm thấy Chatbot.html</h1>"