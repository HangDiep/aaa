import os
from dotenv import load_dotenv
from google import genai

# Load env
ENV_PATH = r"D:\HTML\a - Copy\rag\.env"
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH, override=True)
else:
    load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ No API Key found.")
    exit()

try:
    client = genai.Client(api_key=api_key)
    print("--- Listing Available Models ---")
    for m in client.models.list():
        if "generateContent" in m.supported_generation_methods:
            print(f"- {m.name}")

except Exception as e:
    print(f"Error: {e}")
