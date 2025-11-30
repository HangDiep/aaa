# server.py
# server.py
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import base64
import io
import numpy as np
from pydub import AudioSegment
from faster_whisper import WhisperModel

app = FastAPI()

# === PHỤC VỤ FILE TĨNH ===
app.mount("/static", StaticFiles(directory="static"), name="static")

# === TRANG CHỦ ===
@app.get("/")
async def root():
    return FileResponse("static/index.html")

# === TẢI MODEL WHISPER ===
model = WhisperModel("tiny", device="cpu", compute_type="int8")

# === WEBSOCKET ===
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    buffer = b""
    try:
        while True:
            data = await websocket.receive_text()
            audio_chunk = base64.b64decode(data)
            buffer += audio_chunk

            if len(buffer) > 80000:
                try:
                    audio = AudioSegment.from_file(io.BytesIO(buffer), format="webm")
                    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

                    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                    if audio.channels > 1:
                        samples = samples[::audio.channels]
                    samples /= 32768.0

                    segments, _ = model.transcribe(samples, language="vi", vad_filter=True)
                    text = "".join(seg.text for seg in segments).strip()

                    if text:
                        await websocket.send_text(text)

                    buffer = b""
                except Exception as e:
                    print("Lỗi decode:", e)
                    buffer = b""
    except:
        pass

# Chạy: uvicorn server.py:app --reload