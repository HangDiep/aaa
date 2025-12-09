"""
Enterprise Grade Qdrant Sync Script (Single Collection Architecture)
Tự động embed và push tất cả bảng vào một Collection duy nhất 'knowledge_base'
Tối ưu hóa bộ nhớ: Dùng Batch Processing & Generators để không load hết vào RAM
"""

import os
import sys
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
import gc  # Garbage Collector

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

# Tên Collection duy nhất cho toàn bộ hệ thống
GLOBAL_COLLECTION_NAME = "knowledge_base"
BATCH_SIZE = 32  # Số lượng vector embed mỗi lần (giảm xuống nếu RAM yếu)

def normalize(x: str) -> str:
    if not x: return ""
    return " ".join(str(x).lower().strip().split())

def get_db_connection():
    return sqlite3.connect(FAQ_DB_PATH)

def build_embed_text(row_dict: dict, table_name: str) -> str:
    """
    Tạo text để embed. Ưu tiên các trường quan trọng.
    """
    skip_cols = ["notion_id", "last_updated", "approved"]
    priority_cols = ["name", "title", "question", "ten", "tieu_de", "cau_hoi", "noidung", "content", "answer"]
    
    parts = [f"Chủ đề: {table_name}"]  # Context injection
    
    # Thêm các cột ưu tiên
    for col in priority_cols:
        col_lower = col.lower()
        if col_lower in row_dict and row_dict[col_lower]:
            parts.append(str(row_dict[col_lower]))
            
    # Thêm các cột còn lại
    for col, value in row_dict.items():
        if col not in skip_cols and col.lower() not in priority_cols and value:
            parts.append(f"{col}: {value}")
            
    return normalize(" ".join(parts))

def row_generator(cursor, batch_size=100):
    """Memory-safe generator: Đọc từng cục dữ liệu từ SQLite"""
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        yield rows

def sync_table_to_global_collection(table_name: str, embed_model, client):
    """
    Sync dữ liệu từ bảng table_name vào 'knowledge_base'
    """
    print(f"\n[SYNC] Processing table: {table_name.upper()}")
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Lấy columns
    try:
        cur.execute(f"PRAGMA table_info({table_name})")
        columns_info = cur.fetchall()
        if not columns_info:
            print(f"  ❌ Table '{table_name}' not found.")
            conn.close()
            return
        columns = [col[1] for col in columns_info]
    except Exception as e:
        print(f"  ❌ Error reading table info: {e}")
        conn.close()
        return

    # Sử dụng generator để duyệt qua dữ liệu, KHÔNG load hết vào RAM
    
    # Check if 'approved' column exists
    has_approved = "approved" in [c.lower() for c in columns]
    
    if has_approved:
        sql_query = f"SELECT * FROM {table_name} WHERE approved = 1 OR approved IS NULL"
    else:
        sql_query = f"SELECT * FROM {table_name}"
        
    cur.execute(sql_query)
    
    total_synced = 0
    points_buffer = []

    for rows_chunk in row_generator(cur, batch_size=BATCH_SIZE):
        texts_to_embed = []
        payloads = []
        ids = []

        for row in rows_chunk:
            row_dict = dict(zip(columns, row))
            notion_id = row_dict.get("notion_id")
            
            if not notion_id: continue

            # Build embed text
            text = build_embed_text(row_dict, table_name)
            
            # Tạo payload, QUAN TRỌNG: Thêm source_table
            payload = {k: v for k, v in row_dict.items() if k != "notion_id"}
            payload["source_table"] = table_name  # Metadata dùng để lọc sau này
            
            ids.append(notion_id)
            texts_to_embed.append(text)
            payloads.append(payload)

        # Embed batch này
        if texts_to_embed:
            try:
                embeddings = embed_model.encode(texts_to_embed, normalize_embeddings=True)
                
                # Tạo points
                for i, _id in enumerate(ids):
                    points_buffer.append(PointStruct(
                        id=_id,
                        vector=embeddings[i].tolist(),
                        payload=payloads[i]
                    ))
                
                # Push lên Qdrant ngay khi buffer đủ lớn để giải phóng RAM
                if len(points_buffer) >= BATCH_SIZE:
                    client.upsert(
                        collection_name=GLOBAL_COLLECTION_NAME,
                        points=points_buffer
                    )
                    total_synced += len(points_buffer)
                    print(f"  Saved {len(points_buffer)} items...", end="\r")
                    points_buffer = []  # Clear buffer
                    gc.collect() # Force clear RAM
                    
            except Exception as e:
                print(f"  ⚠️ Error embedding batch: {e}")

    # Push nốt những cái còn sót lại
    if points_buffer:
        client.upsert(
            collection_name=GLOBAL_COLLECTION_NAME,
            points=points_buffer
        )
        total_synced += len(points_buffer)
    
    conn.close()
    print(f"  ✅ Finished syncing {total_synced} items from '{table_name}'")


def init_global_collection(client):
    """Khởi tạo collection duy nhất nếu chưa có"""
    try:
        if not client.collection_exists(GLOBAL_COLLECTION_NAME):
            print(f"Creating global collection: {GLOBAL_COLLECTION_NAME}")
            client.create_collection(
                collection_name=GLOBAL_COLLECTION_NAME,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
            # Tạo Payload Index cho source_table để filter nhanh
            client.create_payload_index(
                collection_name=GLOBAL_COLLECTION_NAME,
                field_name="source_table",
                field_schema="keyword"
            )
            print("✅ Collection & Index created.")
        else:
            print(f"Existing collection found: {GLOBAL_COLLECTION_NAME}")
    except Exception as e:
        print(f"Error checking/creating collection: {e}")

def cleanup_old_collections(client):
    """
    Optional: Xóa các collections cũ lẻ tẻ để dọn rác
    """
    try:
        collections = client.get_collections().collections
        for c in collections:
            if c.name != GLOBAL_COLLECTION_NAME:
                print(f"🗑️ Deleting old fragmented collection: {c.name}")
                client.delete_collection(c.name)
    except Exception as e:
        print(f"Warning cleaning up: {e}")

def main():
    print("🚀 Enterprise Sync Started (Memory Safe Mode)...")
    
    # 1. Load Model
    print("📦 Loading embedding model...")
    try:
        embed_model = SentenceTransformer("BAAI/bge-m3", device="cpu") # Force CPU nếu GPU yếu
    except:
        embed_model = SentenceTransformer("keepitreal/vietnamese-sbert")
    
    # 2. Connect Qdrant
    print(f"🔗 Connecting to Qdrant ({QDRANT_URL})...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # 3. Init Global Collection
    init_global_collection(client)
    
    # 4. Clean up old mess (Theo yêu cầu clean up để chuyển architecture)
    # cleanup_old_collections(client) # Uncomment nếu muốn tự động xóa cái cũ
    
    # 5. Get all tables
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    all_tables = [row[0] for row in cur.fetchall() if not row[0].startswith("sqlite_")]
    conn.close()
    
    specific_table = sys.argv[1] if len(sys.argv) > 1 else None
    
    if specific_table:
        sync_table_to_global_collection(specific_table, embed_model, client)
    else:
        print(f"📋 Found {len(all_tables)} tables to sync.")
        for table in all_tables:
            sync_table_to_global_collection(table, embed_model, client)
            gc.collect() # Dọn RAM sau mỗi bảng
            
    print("\n🎉 GLOBAL SYNC COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
