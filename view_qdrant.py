"""
Script để xem thông tin collections trong Qdrant
"""

import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load .env
ENV_PATH = r"D:\HTML\a - Copy\rag\.env"
try:
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH, override=True)
    else:
        load_dotenv()
except Exception:
    pass

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

print("🔗 Kết nối tới Qdrant...")
if QDRANT_API_KEY:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    client = QdrantClient(url=QDRANT_URL)

print("\n📊 Thông tin Collections:\n")

collections = client.get_collections().collections
for col in collections:
    info = client.get_collection(col.name)
    print(f"📁 Collection: {col.name}")
    print(f"   ├─ Vectors: {info.points_count}")
    print(f"   ├─ Dimension: {info.config.params.vectors.size}")
    print(f"   └─ Distance: {info.config.params.vectors.distance}\n")

print("✅ Tổng số collections:", len(collections))
