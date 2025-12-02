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

    try:
        while True:
            data = await websocket.receive_text()
            audio_chunk = base64.b64decode(data)
            buffer += audio_chunk

            # đủ dữ liệu rồi thì xử lý
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
                        print("🎤 Câu nói:", text)

                        try:
                            answer = process_message(text)
                        except Exception as e:
                            print("Lỗi chatbot:", e)
                            answer = "Hiện tại chatbot gặp lỗi."

                        import json

                        # Gửi tin nhắn của user
                        await websocket.send_text(json.dumps({
                            "sender": "user",
                            "text": text
                        }, ensure_ascii=False))

                        # Gửi tin nhắn của bot
                        await websocket.send_text(json.dumps({
                            "sender": "bot",
                            "text": answer
                        }, ensure_ascii=False))


                    buffer = b""

                except Exception as e:
                    print("Lỗi decode:", e)
                    buffer = b""

    except:
        print("WebSocket đóng.")
        pass

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
