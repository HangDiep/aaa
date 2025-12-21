# ==========================================
# HO TÊN: Đỗ Thị Hồng Điệp
# MSSV: 23103014
# ĐỒ ÁN: Chatbot Dynamic Router - TTN University
# NGÀY NỘP: 21/12/2025
# Copyright © 2025. All rights reserved.
# ==========================================

from fastapi import APIRouter, WebSocket
import json
import base64
import time
import io
import numpy as np
import sys
from pathlib import Path

# Thêm path để import chat_fixed từ thư mục cha
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Load Model
try:
    from faster_whisper import WhisperModel
    from pydub import AudioSegment
    print("🔊 Loading Whisper model in banghiamcuoicung...")
    whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
except Exception as e:
    print(f"❌ Error loading libraries: {e}")
    whisper_model = None

# Import chat logic
try:
    from chat_fixed import process_message
except ImportError:
    try:
        from chat import process_message
    except ImportError:
        process_message = lambda x, **k: "Lỗi: Không tìm thấy module chat."

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🟢 Voice WS connected!")

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
            
            # Xử lý khi đủ silent gap
            if now - last_time > SILENCE_GAP and len(buffer) > MIN_SIZE:
                if whisper_model:
                    try:
                        audio = AudioSegment.from_file(io.BytesIO(buffer), format="webm")
                        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                        samples /= 32768.0
                        
                        segments, info = whisper_model.transcribe(samples, language="vi", vad_filter=True)
                        text = "".join(seg.text for seg in segments).strip()
                        
                        if text:
                            print(f"🎤 Voice User: {text}")
                            # 1. Gửi lại text cho user
                            await websocket.send_text(json.dumps({
                                "sender": "user",
                                "text": text
                            }, ensure_ascii=False))
                            
                            # 2. Hỏi Chatbot
                            answer = process_message(text, session_id="voice_session")
                            
                            # 3. Gửi câu trả lời
                            await websocket.send_text(json.dumps({
                                "sender": "bot",
                                "text": answer
                            }, ensure_ascii=False))
                            
                    except Exception as e:
                       print(f"❌ Transcribe error: {e}")
                else:
                    print("❌ Model chưa load.")

                buffer = b""
            
            # Reset buffer nếu quá dài
            if len(buffer) > MAX_SIZE: buffer = b""
            last_time = now
            
    except Exception as e:
        print(f"🔴 WS closed: {e}")

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=9000, reload=True)
