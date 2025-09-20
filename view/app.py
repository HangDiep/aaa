from fastapi import FastAPI, Form
from fastapi.responses import PlainTextResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from chat_fixed import process_message

app = FastAPI()

# 1) Phục vụ file tĩnh ở /static (không chiếm router gốc)
app.mount("/static", StaticFiles(directory="view", html=True), name="static")
@app.get("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_probe():
    return PlainTextResponse("", status_code=204)
# 2) Trang chủ: bật thẳng Chatbot.html (đổi tên nếu bạn dùng tên khác)
@app.get("/")
@app.get("/")
def home():
    return FileResponse(os.path.join("view","Chatbot.html"),
                        media_type="text/html; charset=utf-8")


# 3) API nhận form POST từ Chatbot.html


@app.post("/chat", response_class=PlainTextResponse)
def chat(message: str = Form(...)):
    return process_message(message)

# (tuỳ chọn) favicon để khỏi thấy 404 trong console
@app.get("/favicon.ico")
def favicon():
    ico = os.path.join(os.getcwd(), "favicon.ico")
    if os.path.exists(ico):
        return FileResponse(ico, media_type="image/x-icon")
    return PlainTextResponse("", status_code=204)
