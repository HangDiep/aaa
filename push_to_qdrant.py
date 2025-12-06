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

def ensure_collection(name: str, vector_size: int = 1024):
    """Create collection if it doesn't exist (don't delete existing)"""
    try:
        client.get_collection(collection_name=name)
        print(f"  ✅ Collection '{name}' already exists")
    except:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"  ✅ Created new collection: {name}")

def get_existing_points(collection_name: str) -> dict:
    """
    Get all existing points from Qdrant collection
    Returns: {notion_id: last_updated}
    """
    try:
        # Scroll through all points in collection
        points_map = {}
        offset = None
        
        while True:
            result = client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            points, next_offset = result
            
            for point in points:
                notion_id = str(point.id)
                last_updated = point.payload.get("last_updated", "")
                points_map[notion_id] = last_updated
            
            if next_offset is None:
                break
            offset = next_offset
        
        return points_map
    except Exception as e:
        print(f"  ⚠️ Error getting existing points: {e}")
        return {}

def sync_collection_incremental(collection_name: str, data_rows: list, embed_model):
    """
    Incrementally sync SQLite data to Qdrant
    
    Args:
        collection_name: Qdrant collection name
        data_rows: List of tuples (notion_id, embed_text, ...fields..., last_updated)
        embed_model: SentenceTransformer model
    """
    print(f"\n[SYNC] {collection_name.upper()} Collection...")
    
    # Ensure collection exists
    ensure_collection(collection_name)
    
    # Get existing points
    existing_points = get_existing_points(collection_name)
    print(f"  📊 Existing points in Qdrant: {len(existing_points)}")
    
    # Build SQLite data map
    sqlite_map = {row[0]: row for row in data_rows}  # {notion_id: full_row}
    sqlite_ids = set(sqlite_map.keys())
    existing_ids = set(existing_points.keys())
    
    # Find changes
    new_ids = sqlite_ids - existing_ids
    deleted_ids = existing_ids - sqlite_ids
    potential_updates = sqlite_ids & existing_ids
    
    # Check for actual updates (timestamp comparison)
    updated_ids = set()
    for notion_id in potential_updates:
        sqlite_updated = sqlite_map[notion_id][-1]  # last_updated is last field
        qdrant_updated = existing_points[notion_id]
        if sqlite_updated != qdrant_updated:
            updated_ids.add(notion_id)
    
    print(f"  🆕 New records: {len(new_ids)}")
    print(f"  🔄 Updated records: {len(updated_ids)}")
    print(f"  🗑️  Deleted records: {len(deleted_ids)}")
    
    # Process new + updated records
    to_upsert_ids = new_ids | updated_ids
    if to_upsert_ids:
        to_upsert_rows = [sqlite_map[nid] for nid in to_upsert_ids]
        embed_texts = [row[1] for row in to_upsert_rows]  # embed_text is index 1
        
        print(f"  🧠 Embedding {len(embed_texts)} records...")
        embeddings = embed_model.encode(embed_texts, normalize_embeddings=True)
        
        # Build points based on collection type
        points = []
        for i, row in enumerate(to_upsert_rows):
            notion_id = row[0]
            
            if collection_name == "faq":
                # (notion_id, embed_text, question, answer, category, last_updated)
                payload = {
                    "notion_id": notion_id,
                    "question": row[2] or "",
                    "answer": row[3] or "",
                    "category": row[4] or "",
                    "last_updated": row[5]
                }
            elif collection_name == "books":
                # (notion_id, embed_text, name, author, year, quantity, status, major, last_updated)
                payload = {
                    "notion_id": notion_id,
                    "name": row[2],
                    "author": row[3],
                    "year": row[4],
                    "quantity": row[5],
                    "status": row[6],
                    "major": row[7] or "Chung",
                    "last_updated": row[8]
                }
            elif collection_name == "majors":
                # (notion_id, embed_text, name, major_id, description)
                payload = {
                    "notion_id": notion_id,
                    "name": row[2],
                    "major_id": row[3],
                    "description": row[4] or "Đang cập nhật"
                }
            
            points.append(PointStruct(
                id=notion_id,  # Use notion_id as point ID
                vector=embeddings[i].tolist(),
                payload=payload
            ))
        
        # Upsert in batches
        batch_size = 20
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            client.upsert(collection_name=collection_name, points=batch)
            if len(points) > batch_size:
                print(f"  ⏳ Upserted {min(i+batch_size, len(points))}/{len(points)} points...")
        
        print(f"  ✅ Upserted {len(points)} points")
    
    # Delete removed records
    if deleted_ids:
        client.delete(
            collection_name=collection_name,
            points_selector=list(deleted_ids)
        )
        print(f"  ✅ Deleted {len(deleted_ids)} points")
    
    if not to_upsert_ids and not deleted_ids:
        print(f"  ✨ No changes detected - collection is up to date!")

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
cur.execute("""
    SELECT notion_id, question, answer, category, last_updated 
    FROM faq 
    WHERE approved = 1 OR approved IS NULL
""")
faq_rows = cur.fetchall()
FAQ_DATA = [(row[0], normalize(f"{row[3] or ''}: {row[2] or ''}"), row[1], row[2], row[3], row[4]) 
            for row in faq_rows if row[0] is not None]  # Skip rows with None notion_id

# BOOKS
cur.execute("""
    SELECT b.notion_id, b.name, b.author, b.year, b.quantity, b.status, 
           m.name, b.last_updated
    FROM books b 
    LEFT JOIN majors m ON b.major_id = m.major_id
""")
book_rows = cur.fetchall()
BOOK_DATA = [(row[0], normalize(f"sách {row[1]}. tác giả {row[2]}. ngành {row[6] or ''}"), 
              row[1], row[2], row[3], row[4], row[5], row[6], row[7])
             for row in book_rows if row[0] is not None]  # Skip rows with None notion_id

# MAJORS
cur.execute("""
    SELECT notion_id, name, major_id, description 
    FROM majors
""")
major_rows = cur.fetchall()
MAJOR_DATA = [(row[0], normalize(f"ngành {row[1]}. mã {row[2]}. {row[3] or ''}"), 
               row[1], row[2], row[3])
              for row in major_rows if row[0] is not None]  # Skip rows with None notion_id

conn.close()

print(f"  ✅ FAQ: {len(FAQ_DATA)} rows")
print(f"  ✅ BOOKS: {len(BOOK_DATA)} rows")
print(f"  ✅ MAJORS: {len(MAJOR_DATA)} rows")

# ============================================
#  INCREMENTAL SYNC TO QDRANT
# ============================================
print("\n🧠 Đang sync embeddings vào Qdrant (incremental)...")

# Sync each collection
sync_collection_incremental("faq", FAQ_DATA, embed_model)
sync_collection_incremental("books", BOOK_DATA, embed_model)
sync_collection_incremental("majors", MAJOR_DATA, embed_model)

print("\n🎉 HOÀN TẤT! Embeddings đã được sync vào Qdrant.")
print("👉 Bây giờ bạn có thể chạy chat.py (sẽ đọc từ Qdrant)")
