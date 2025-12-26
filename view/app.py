# view/app.py
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
import sys
import os
import uuid
import asyncio

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import logics
from chat_fixed import process_message
from sync_dynamic import scan_new_databases, router as sync_router
try:
    from banghiamcuoicung.server import router as voice_router
except ImportError:
    voice_router = None

app = FastAPI(title="Library Chatbot - TTN University")

# Mount static files
app.mount("/static", StaticFiles(directory="view"), name="static")

# Include Routers
app.include_router(sync_router)
if voice_router:
    app.include_router(voice_router)
    print("✅ Voice WebSocket router included at /ws")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= BACKGROUND SYNC (3-MIN AUTO SCAN) =================
@app.on_event("startup")
async def startup_event():
    """Khởi động quét Notion tự động mỗi 3 phút."""
    asyncio.create_task(run_auto_scan_loop())

async def run_auto_scan_loop():
    interval = int(os.getenv("SYNC_INTERVAL_SECONDS", 180))
    print(f"⏰ [Server] Auto-Scan started (Every {interval}s)")
    
    while True:
        try:
            print(f"\n⏰ [Auto-Scan] Triggering scheduled scan...")
            await scan_new_databases()
        except Exception as e:
            print(f"❌ [Auto-Scan] Error: {e}")
        
        await asyncio.sleep(interval)

# ================= ROUTES =================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(message: str = Form(""), session_id: str = Form("default"), image: UploadFile = File(None)):
    image_path = None
    if image and image.filename:
        suffix = Path(image.filename).suffix.lower()
        safe_filename = f"{uuid.uuid4()}{suffix}"
        os.makedirs("temp", exist_ok=True)
        image_path = Path("temp") / safe_filename
        
        content = await image.read()
        with open(image_path, "wb") as f:
            f.write(content)
        
        print(f"[UPLOAD] Đã lưu ảnh → {image_path} | Session: {session_id}")

    try:
        answer = process_message(
            message.strip(), 
            session_id=session_id, 
            image_path=str(image_path) if image_path else None
        )
    finally:
        if image_path and image_path.exists():
            try:
                image_path.unlink()
            except:
                pass

    return {"answer": answer}

@app.get("/", response_class=HTMLResponse)
def home():
    file_path = Path(__file__).parent / "index.html"
    if not file_path.exists():
        return "<h1>Không tìm thấy index.html</h1>"
    return file_path.read_text(encoding="utf-8")

@app.get("/chatbot", response_class=HTMLResponse)
def chatbot_page():
    file_path = Path(__file__).parent / "Chatbot.html"
    if not file_path.exists():
        return "<h1>Không tìm thấy Chatbot.html</h1>"
    return file_path.read_text(encoding="utf-8")

@app.post("/reload-config")
def reload_config():
    try:
        from chat_dynamic_router import trigger_config_reload
        collections = trigger_config_reload()
        return {"status": "ok", "collections": list(collections.keys())}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/ping")
def ping():
    return {"msg": "pong"}
