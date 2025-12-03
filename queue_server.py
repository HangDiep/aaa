# queue_server.py (ví dụ)
import asyncio
import time
from fastapi import FastAPI
from pydantic import BaseModel
import httpx  # để call Groq

app = FastAPI()

# ====== CẤU TRÚC JOB ======
class Job:
    def __init__(self, question: str):
        self.question = question
        self.future: asyncio.Future = asyncio.get_event_loop().create_future()

# Hàng đợi chung
job_queue: asyncio.Queue[Job] = asyncio.Queue()

# ====== CẤU HÌNH GỌI GROQ ======
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = "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"  # nhớ để vào .env

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}

# Hàm gọi Groq với retry + delay khi 429
async def call_groq_with_retry(question: str, max_retries: int = 5):
    backoff = 1.5  # hệ số nhân delay
    delay = 1.0    # delay ban đầu (giây)

    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(max_retries):
            resp = await client.post(
                GROQ_API_URL,
                headers=HEADERS,
                json={
                    "model": "llama-3.1-70b-versatile",  # ví dụ
                    "messages": [
                        {"role": "user", "content": question}
                    ],
                    "temperature": 0.2,
                },
            )

            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]

            # Nếu bị 429 → chờ rồi thử lại
            if resp.status_code == 429:
                print(f"⚠ 429 từ Groq. attempt={attempt+1}")
                await asyncio.sleep(delay)
                delay *= backoff
                continue

            # Lỗi khác 429 → break luôn
            print(f"❌ Lỗi Groq: {resp.status_code} - {resp.text}")
            break

    # Nếu retry hết mà vẫn fail
    return "Xin lỗi, hệ thống đang quá tải. Bạn vui lòng thử lại sau."

# ====== WORKER XỬ LÝ HÀNG ĐỢI ======
async def worker_loop():
    print("✅ Worker hàng đợi Groq đã chạy")
    last_call_time = 0.0

    while True:
        job: Job = await job_queue.get()  # chờ có job mới
        now = time.time()

        # THROTTLE: đảm bảo mỗi request cách nhau ít nhất 2 giây
        min_interval = 2.0  # GIÁ TRỊ QUAN TRỌNG: 2s ~ 30 req/phút
        if now - last_call_time < min_interval:
            wait = min_interval - (now - last_call_time)
            print(f"⏳ Chờ {wait:.2f}s để không vượt limit Groq")
            await asyncio.sleep(wait)

        # Gọi Groq
        answer = await call_groq_with_retry(job.question)
        last_call_time = time.time()

        # Ghi kết quả vào future để endpoint trả về cho user
        if not job.future.done():
            job.future.set_result(answer)

        job_queue.task_done()

# ====== CHẠY WORKER KHI STARTUP ======
@app.on_event("startup")
async def startup_event():
    # Tạo 1 worker chạy nền
    asyncio.create_task(worker_loop())

# ====== API NHẬN CÂU HỎI TỪ USER ======
class AskRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask(req: AskRequest):
    # Tạo job
    job = Job(question=req.question)
    # Đưa vào hàng đợi
    await job_queue.put(job)
    # Chờ worker xử lý xong
    answer = await job.future
    return {"answer": answer}
