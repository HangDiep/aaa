# ======================
#  LIBRARY CHATBOT API (FULL CHAT + OCR + WEBSOCKET VOICE)
# ======================

from fastapi import FastAPI, Form, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import sys, json, base64, io, time
from fastapi.responses import HTMLResponse
import numpy as np
from pydub import AudioSegment
from faster_whisper import WhisperModel

# -----------------------
#  INIT APP
# -----------------------
app = FastAPI(title="Library Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -----------------------
#  IMPORT CHAT LOGIC
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from chat_fixed import process_message   # dùng cho Chat + Voice

# -----------------------
#  LOAD WHISPER MODEL
# -----------------------
print("🔊 Loading Whisper model...")
model = WhisperModel("tiny", device="cpu")

# -----------------------
#  API: HEALTH CHECK
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------
#  API: CHAT
# -----------------------
@app.post("/chat")
def chat(message: str = Form(...)):
    return {"answer": process_message(message)}

# -----------------------
#  DUMMY SEARCH / INVENTORY
# -----------------------
@app.get("/search")
def search(q: str):
    return [{"answer": "Giờ mở cửa: 7:30 - 17:00, Thứ 2–Thứ 6."}]

@app.get("/inventory")
def inventory(book_name: str):
    return [{"name": book_name, "author": "N/A", "year": "?", "quantity": 3, "status": "available"}]

# ============================================================
#  WEBSOCKET: VOICE RECOGNITION (REALTIME)
# ============================================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🟢 WS connected!")

    buffer = b""
    last_time = None

    SILENCE_GAP = 0.55
    MIN_SIZE = 4000
    MAX_SIZE = 150000

    while True:
        try:
            # nhận base64 từ front-end
            data = await websocket.receive_text()
            chunk = base64.b64decode(data)
            buffer += chunk

            now = time.time()
            if last_time is None:
                last_time = now

            # detect im lặng
            if now - last_time > SILENCE_GAP and len(buffer) > MIN_SIZE:
                try:
                    audio = AudioSegment.from_file(io.BytesIO(buffer), format="webm")
                    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

                    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                    samples /= 32768.0

                    segments, _ = model.transcribe(samples, language="vi", vad_filter=True)
                    text = "".join(seg.text for seg in segments).strip()

                    if text:
                        print("🎤 User:", text)

                        answer = process_message(text)

                        await websocket.send_text(json.dumps({
                            "sender": "user",
                            "text": text
                        }, ensure_ascii=False))

                        await websocket.send_text(json.dumps({
                            "sender": "bot",
                            "text": answer
                        }, ensure_ascii=False))

                except Exception as e:
                    print("❌ decode error:", e)

                buffer = b""

            if len(buffer) > MAX_SIZE:
                buffer = b""

            last_time = now

        except Exception as e:
            print("🔴 WS closed:", e)
            break

# ============================================================
#  STATIC FILES + HTML UI
# ============================================================
STATIC_DIR = Path(__file__).resolve().parent

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    file_path = STATIC_DIR / "Chatbot.html"
    if not file_path.exists():
        return "<h1>❌ Không tìm thấy Chatbot.html</h1>"
    return file_path.read_text(encoding="utf-8")

@app.get("/ping")
def ping():
    return {"msg": "pong"}
