"""
Qdrant Incremental Sync Script (Single Collection Architecture)

- Mỗi bảng trong SQLite tương ứng với source_table trong Qdrant.
- Khi sync một bảng:
    + Upsert tất cả row hiện có trong SQLite (chỉ row approved = 1, trừ bảng 'nganh').
    + Lấy danh sách ID hiện đang có trong Qdrant cho bảng đó.
    + Xoá các ID đã có trong Qdrant nhưng không còn trong SQLite (đã xoá / unapproved trên Notion).
=> Không cần xoá sạch cả bảng trong Qdrant, chỉ xoá đúng record "mồ côi".
"""

import os
import sys
import sqlite3
import gc  # Garbage Collector

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from dotenv import load_dotenv

# ==========================
#  Load env
# ==========================

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

GLOBAL_COLLECTION_NAME = "knowledge_base"
BATCH_SIZE = 32  # batch embed


# ==========================
#  Helper
# ==========================

def normalize(x: str) -> str:
    if not x:
        return ""
    return " ".join(str(x).lower().strip().split())

def get_table_description_from_sqlite(table_name: str) -> str:
    try:
        conn = sqlite3.connect(FAQ_DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT description FROM collections_config WHERE name=?", (table_name,))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else f"Mô tả bảng {table_name}"
    except:
        return f"Mô tả bảng {table_name}"

def get_db_connection():
    return sqlite3.connect(FAQ_DB_PATH)


def build_embed_text(row_dict: dict, table_name: str) -> str:
    """
    Tạo text để embed. Ưu tiên các trường quan trọng.
    """
    skip_cols = ["notion_id", "last_updated", "approved"]
    priority_cols = [
        "name",
        "title",
        "question",
        "ten",
        "tieu_de",
        "cau_hoi",
        "noidung",
        "content",
        "answer",
    ]

    parts = [f"Chủ đề: {table_name}"]

    for col in priority_cols:
        col_lower = col.lower()
        if col_lower in row_dict and row_dict[col_lower]:
            parts.append(str(row_dict[col_lower]))

    for col, value in row_dict.items():
        if col not in skip_cols and col.lower() not in priority_cols and value:
            parts.append(f"{col}: {value}")

    return normalize(" ".join(parts))


def row_generator(cursor, batch_size=100):
    """Đọc từng cục dữ liệu từ SQLite, tránh full RAM."""
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        yield rows


def get_existing_ids_in_qdrant(client: QdrantClient, table_name: str):
    """
    Lấy toàn bộ ID (notion_id) hiện đang có trong Qdrant cho source_table = table_name.
    Dùng scroll với filter.
    """
    existing_ids = set()
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=GLOBAL_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="source_table",
                        match=MatchValue(value=table_name),
                    )
                ]
            ),
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )

        for p in points:
            existing_ids.add(str(p.id))

        if offset is None:
            break

    print(f"  🔎 Qdrant currently has {len(existing_ids)} ids for table '{table_name}'")
    return existing_ids


# ==========================
#  Qdrant Sync per Table
# ==========================

def sync_table_to_global_collection(table_name: str, embed_model, client: QdrantClient):
    """
    Incremental sync cho 1 bảng:

    1. Đọc dữ liệu hiện tại từ SQLite:
        - Nếu có cột approved & bảng != 'nganh' → chỉ lấy approved = 1
        - Ngược lại → lấy tất cả.
    2. Upsert embedding cho tất cả row đó vào Qdrant.
    3. Lấy danh sách ID đang có trong Qdrant (theo source_table).
    4. Xoá các ID có trong Qdrant nhưng không còn trong SQLite.
    """

    print(f"\n[SYNC] Processing table: {table_name.upper()}")

    # 1. Lấy danh sách ID hiện đang có trong Qdrant cho bảng này
    existing_ids = get_existing_ids_in_qdrant(client, table_name)

    # 2. Đọc dữ liệu từ SQLite
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute(f"PRAGMA table_info({table_name})")
        columns_info = cur.fetchall()
        if not columns_info:
            print(f"  ❌ Table '{table_name}' not found in SQLite.")
            conn.close()
            return
        columns = [col[1] for col in columns_info]
    except Exception as e:
        print(f"  ❌ Error reading table info for '{table_name}': {e}")
        conn.close()
        return

    lower_columns = [c.lower() for c in columns]
    has_approved = "approved" in lower_columns
    lower_table_name = table_name.lower()

    if has_approved and lower_table_name != "nganh":
        sql_query = f"SELECT * FROM {table_name} WHERE approved = 1"
    else:
        sql_query = f"SELECT * FROM {table_name}"

    print(f"  🔎 SQL: {sql_query}")
    try:
        cur.execute(sql_query)
    except Exception as e:
        print(f"  ❌ Error executing query on '{table_name}': {e}")
        conn.close()
        return

    total_synced = 0
    points_buffer = []
    sqlite_ids = set()  # lưu lại toàn bộ notion_id hiện có trong SQLite cho bảng này

    for rows_chunk in row_generator(cur, batch_size=BATCH_SIZE):
        texts_to_embed = []
        payloads = []
        ids = []

        for row in rows_chunk:
            row_dict = dict(zip(columns, row))
            notion_id = row_dict.get("notion_id")

            if not notion_id:
                continue

            notion_id_str = str(notion_id)
            sqlite_ids.add(notion_id_str)

            text = build_embed_text(row_dict, table_name)

            payload = {k: v for k, v in row_dict.items() if k != "notion_id"}
            payload["source_table"] = table_name
            # 🔥 NEW: Gắn mô tả bảng vào Qdrant
            description = get_table_description_from_sqlite(table_name)
            payload["table_description"] = description
            ids.append(notion_id_str)
            texts_to_embed.append(text)
            payloads.append(payload)

        if texts_to_embed:
            try:
                embeddings = embed_model.encode(
                    texts_to_embed, normalize_embeddings=True
                )

                for i, _id in enumerate(ids):
                    points_buffer.append(
                        PointStruct(
                            id=_id,
                            vector=embeddings[i].tolist(),
                            payload=payloads[i],
                        )
                    )

                if len(points_buffer) >= BATCH_SIZE:
                    client.upsert(
                        collection_name=GLOBAL_COLLECTION_NAME,
                        points=points_buffer,
                    )
                    total_synced += len(points_buffer)
                    print(f"  💾 Upserted {len(points_buffer)} items...", end="\r")
                    points_buffer = []
                    gc.collect()

            except Exception as e:
                print(f"  ⚠️ Error embedding batch: {e}")

    if points_buffer:
        client.upsert(
            collection_name=GLOBAL_COLLECTION_NAME,
            points=points_buffer,
        )
        total_synced += len(points_buffer)

    conn.close()
    print(f"\n  ✅ Finished upserting {total_synced} items from '{table_name}'")

    # 3. Xoá các ID "mồ côi" trong Qdrant (có trong Qdrant nhưng không còn trong SQLite)
    ids_to_delete = existing_ids - sqlite_ids
    if ids_to_delete:
        print(
            f"  🗑️ Deleting {len(ids_to_delete)} obsolete points in Qdrant for table '{table_name}'..."
        )
        try:
            client.delete(
                collection_name=GLOBAL_COLLECTION_NAME,
                points_selector=list(ids_to_delete),  # qdrant-client nhận list id trực tiếp
            )
            print("  ✅ Obsolete points deleted from Qdrant.")
        except Exception as e:
            print(f"  ⚠️ Error deleting obsolete points: {e}")
    else:
        print("  ✔ No obsolete points to delete in Qdrant.")

    gc.collect()


# ==========================
#  Collection Init & Cleanup
# ==========================

def init_global_collection(client: QdrantClient):
    """Khởi tạo collection duy nhất nếu chưa có"""
    try:
        if not client.collection_exists(GLOBAL_COLLECTION_NAME):
            print(f"Creating global collection: {GLOBAL_COLLECTION_NAME}")
            client.create_collection(
                collection_name=GLOBAL_COLLECTION_NAME,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
            client.create_payload_index(
                collection_name=GLOBAL_COLLECTION_NAME,
                field_name="source_table",
                field_schema="keyword",
            )
            print("✅ Collection & Index created.")
        else:
            print(f"Existing collection found: {GLOBAL_COLLECTION_NAME}")
    except Exception as e:
        print(f"Error checking/creating collection: {e}")


def cleanup_old_collections(client: QdrantClient):
    """
    Optional: Xóa các collections cũ lẻ tẻ để dọn rác (nếu trước đây dùng nhiều collection)
    """
    try:
        collections = client.get_collections().collections
        for c in collections:
            if c.name != GLOBAL_COLLECTION_NAME:
                print(f"🗑️ Deleting old fragmented collection: {c.name}")
                client.delete_collection(c.name)
    except Exception as e:
        print(f"Warning cleaning up: {e}")


# ==========================
#  Main
# ==========================

def main():
    print("🚀 Incremental Qdrant Sync Started...")

    # 1. Load Model
    print("📦 Loading embedding model...")
    try:
        embed_model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    except Exception:
        embed_model = SentenceTransformer("keepitreal/vietnamese-sbert")

    # 2. Connect Qdrant
    print(f"🔗 Connecting to Qdrant ({QDRANT_URL})...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # 3. Init Global Collection
    init_global_collection(client)

    # 4. Optional: cleanup collections cũ
    # cleanup_old_collections(client)

    # 5. Sync
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    all_tables = [
        row[0] for row in cur.fetchall() if not row[0].startswith("sqlite_")
    ]
    conn.close()

    # Nếu script được gọi với tên bảng → chỉ sync bảng đó
    specific_table = sys.argv[1] if len(sys.argv) > 1 else None

    if specific_table:
        print(f"📌 Running in single-table mode: {specific_table}")
        sync_table_to_global_collection(specific_table, embed_model, client)
    else:
        print(f"📋 Found {len(all_tables)} tables to sync.")
        for table in all_tables:
            sync_table_to_global_collection(table, embed_model, client)
            gc.collect()

    print("\n🎉 INCREMENTAL SYNC COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
