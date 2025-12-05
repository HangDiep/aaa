"""
Script để push embeddings từ SQLite vào Qdrant Vector Database
Chạy 1 lần để migrate data, sau đó chat.py sẽ đọc từ Qdrant (nhanh hơn)
"""

import os
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
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

FAQ_DB_PATH = os.getenv("FAQ_DB_PATH", r"D:\HTML\a - Copy\faq.db")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

print("🚀 Bắt đầu migrate embeddings vào Qdrant...")

# ============================================
#  LOAD EMBEDDING MODEL
# ============================================
print("📦 Đang tải model embedding (BAAI/bge-m3)...")
try:
    embed_model = SentenceTransformer("BAAI/bge-m3")
except Exception as e:
    print(f"⚠ Lỗi load model: {e}")
    print("Đang dùng fallback model (keepitreal/vietnamese-sbert)...")
    embed_model = SentenceTransformer("keepitreal/vietnamese-sbert")

# ============================================
#  CONNECT TO QDRANT
# ============================================
print(f"🔗 Kết nối tới Qdrant ({QDRANT_URL})...")
if QDRANT_API_KEY:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print("  ✅ Đã xác thực với API key")
else:
    client = QdrantClient(url=QDRANT_URL)
    print("  ⚠️ Kết nối không có API key (localhost mode)")

# ============================================
#  HELPER FUNCTIONS
# ============================================
def normalize(x: str) -> str:
    return " ".join(x.lower().strip().split())

def create_collection(name: str, vector_size: int = 1024):
    """Tạo collection trong Qdrant (xóa nếu đã tồn tại)"""
    try:
        client.delete_collection(collection_name=name)
        print(f"  ♻️  Đã xóa collection cũ: {name}")
    except:
        pass
    
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"  ✅ Đã tạo collection: {name}")

# ============================================
#  LOAD DATA FROM SQLITE
# ============================================
print("\n📂 Đang đọc dữ liệu từ faq.db...")

if not os.path.exists(FAQ_DB_PATH):
    print(f"❌ Không tìm thấy file {FAQ_DB_PATH}")
    exit(1)

conn = sqlite3.connect(FAQ_DB_PATH)
cur = conn.cursor()

# FAQ
cur.execute("SELECT question, answer, category FROM faq WHERE approved = 1 OR approved IS NULL")
faq_rows = cur.fetchall()
FAQ_TEXTS = [normalize(f"{cat or ''}: {a or ''}") for _, a, cat in faq_rows]

# BOOKS
cur.execute("""
    SELECT b.name, b.author, b.year, b.quantity, b.status, m.name
    FROM books b LEFT JOIN majors m ON b.major_id = m.major_id
""")
book_rows = cur.fetchall()
BOOK_TEXTS = [normalize(f"sách {n}. tác giả {a}. ngành {m or ''}") for n, a, _, _, _, m in book_rows]

# MAJORS
cur.execute("SELECT name, major_id, description FROM majors")
major_rows = cur.fetchall()
MAJOR_TEXTS = [normalize(f"ngành {n}. mã {mid}. {desc or ''}") for n, mid, desc in major_rows]

conn.close()

print(f"  ✅ FAQ: {len(faq_rows)} rows")
print(f"  ✅ BOOKS: {len(book_rows)} rows")
print(f"  ✅ MAJORS: {len(major_rows)} rows")

# ============================================
#  EMBEDDING & PUSH TO QDRANT
# ============================================
print("\n🧠 Đang tạo embeddings và push vào Qdrant...")

# FAQ Collection
print("\n[1/3] FAQ Collection...")
create_collection("faq", vector_size=1024)
if FAQ_TEXTS:
    faq_emb = embed_model.encode(FAQ_TEXTS, normalize_embeddings=True)
    points = [
        PointStruct(
            id=i,
            vector=faq_emb[i].tolist(),
            payload={
                "question": faq_rows[i][0] or "",
                "answer": faq_rows[i][1] or "",
                "category": faq_rows[i][2] or ""
            }
        )
        for i in range(len(faq_rows))
    ]
    client.upsert(collection_name="faq", points=points)
    print(f"  ✅ Đã push {len(points)} vectors vào collection 'faq'")

# BOOKS Collection
print("\n[2/3] BOOKS Collection...")
create_collection("books", vector_size=1024)
if BOOK_TEXTS:
    book_emb = embed_model.encode(BOOK_TEXTS, normalize_embeddings=True)
    points = [
        PointStruct(
            id=i,
            vector=book_emb[i].tolist(),
            payload={
                "name": book_rows[i][0],
                "author": book_rows[i][1],
                "year": book_rows[i][2],
                "quantity": book_rows[i][3],
                "status": book_rows[i][4],
                "major": book_rows[i][5] or "Chung"
            }
        )
        for i in range(len(book_rows))
    ]
    # Push theo batch để tránh timeout
    batch_size = 20
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(collection_name="books", points=batch)
        print(f"  ⏳ Đã push {min(i+batch_size, len(points))}/{len(points)} vectors...")
    print(f"  ✅ Đã push {len(points)} vectors vào collection 'books'")

# MAJORS Collection
print("\n[3/3] MAJORS Collection...")
create_collection("majors", vector_size=1024)
if MAJOR_TEXTS:
    major_emb = embed_model.encode(MAJOR_TEXTS, normalize_embeddings=True)
    points = [
        PointStruct(
            id=i,
            vector=major_emb[i].tolist(),
            payload={
                "name": major_rows[i][0],
                "major_id": major_rows[i][1],
                "description": major_rows[i][2] or "Đang cập nhật"
            }
        )
        for i in range(len(major_rows))
    ]
    client.upsert(collection_name="majors", points=points)
    print(f"  ✅ Đã push {len(points)} vectors vào collection 'majors'")

print("\n🎉 HOÀN TẤT! Embeddings đã được lưu vào Qdrant.")
print("👉 Bây giờ bạn có thể chạy chat.py (sẽ đọc từ Qdrant thay vì embedding lại)")
