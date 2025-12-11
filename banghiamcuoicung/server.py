from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import base64
import io
import numpy as np
from pydub import AudioSegment
from faster_whisper import WhisperModel
import os, sys

# ===== THÊM ĐƯỜNG DẪN TỚI THƯ MỤC CHA (chứa chat_fixed.py) =====
# server.py:  D:\HTML\a - Copy\banghiamcuoicung\server.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # ...\a - Copy\banghiamcuoicung
ROOT_DIR = os.path.dirname(BASE_DIR)                    # ...\a - Copy

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from chat_fixed import process_message   # giờ mới import được

app = FastAPI()



# === PHỤC VỤ STATIC ===
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# === MODEL STT ===
model = WhisperModel("tiny", device="cpu", compute_type="int8")

# === WEBSOCKET ===
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    buffer = b""
    last_time = None

    import time, json

    SILENCE_GAP = 0.55     # user ngừng nói 0.55s
    MIN_SIZE = 4000        # tối thiểu dữ liệu
    MAX_SIZE = 150000      # tránh tràn bộ nhớ

    while True:
        try:
            data = await websocket.receive_text()
            chunk = base64.b64decode(data)
            buffer += chunk

            now = time.time()
            if last_time is None:
                last_time = now

            # Nếu user im lặng → xử lý
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

                        # gửi text user
                        await websocket.send_text(json.dumps({
                            "sender": "user",
                            "text": text
                        }, ensure_ascii=False))

                        # gửi text bot
                        await websocket.send_text(json.dumps({
                            "sender": "bot",
                            "text": answer
                        }, ensure_ascii=False))

                except Exception as e:
                    print("Lỗi decode:", e)

                buffer = b""

            if len(buffer) > MAX_SIZE:
                buffer = b""

            last_time = now

        except Exception:
            print("WebSocket đóng")
            break


# chạy:
# uvicorn server:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=9000,
        reload=True
    )
