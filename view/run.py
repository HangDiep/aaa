import subprocess
import threading
import time
import webbrowser
import os

def start_server():
    subprocess.run(["uvicorn", "app:app", "--reload", "--port", "8000"])

def open_browser():
    time.sleep(2)  # đợi server khởi động
    webbrowser.open("http://127.0.0.1:8000")

# Chạy server ở thread riêng
threading.Thread(target=start_server, daemon=True).start()
open_browser()

print("Đang chạy Thư viện + Chatbot tại http://127.0.0.1:8000")
print("Nhấn Ctrl+C để dừng")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nĐã dừng server!")