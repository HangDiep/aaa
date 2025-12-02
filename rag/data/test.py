from google import genai
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

# Khởi tạo client với key từ môi trường
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Gửi câu hỏi
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello! Explain AI in one sentence."
)

print(response.text)
