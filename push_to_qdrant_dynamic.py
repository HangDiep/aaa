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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "rag", ".env")

try:
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH, override=True)
    else:
        load_dotenv()
except Exception:
    pass

FAQ_DB_PATH = os.getenv("FAQ_DB_PATH", os.path.join(BASE_DIR, "faq.db"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

GLOBAL_COLLECTION_NAME = "knowledge_base"
BATCH_SIZE = 32  # batch embed

EXCLUDED_TABLES = {"questions_log", "sync_meta", "collections_config", "conversations"}


# ==========================
#  Helper
# ==========================

def normalize(x: str) -> str:
    if not x:
        return ""
    return " ".join(str(x).lower().strip().split())


def flatten_recursive(value):
    """
    Hàm đệ quy làm phẳng dữ liệu từ kết quả Notion API.
    Không cần biết tên cột, chỉ nhìn cấu trúc dữ liệu.
    
    Quy tắc:
    - Dict có 'number' -> lấy number
    - Dict có 'select' -> lấy name
    - Dict có 'multi_select' -> lấy list name
    - List -> đệ quy từng phần tử
    - Chuỗi JSON (str) bắt đầu bằng '[' hoặc '{' -> thử parse rồi đệ quy
    """
    import json
    
    if value is None:
        return None
        
    # Trường hợp giá trị 'type': 'number', 'number': 123... (Notion format)
    if isinstance(value, dict):
        if "type" in value:
            t = value.get("type")
            if t in value: # e.g. "type": "number", "number": ...
                return flatten_recursive(value[t])
        
        # Xử lý các object cụ thể
        if "name" in value: # Select, Multi-select value
            return value["name"]
        if "start" in value: # Date
            return value["start"]
        if "email" in value:
            return value["email"]
        if "url" in value:
            return value["url"]
        if "phone_number" in value:
            return value["phone_number"]
        if "number" in value: # Direct number object
            return value["number"]
        if "content" in value: # Text
            return value["content"]
        
        # Nếu là dict thường, duyệt qua các key (nhưng Notion thường nested sâu, 
        # nên tốt rhat là trả về string nếu không match pattern nào)
        return str(value)

    # Trường hợp list (Relation, Multi-select, People...)
    if isinstance(value, list):
        return [flatten_recursive(v) for v in value]
    
    # Trường hợp chuỗi nhưng lại là JSON (do SQLite lưu JSON text)
    if isinstance(value, str):
        value = value.strip()
        if (value.startswith("{") and value.endswith("}")) or \
           (value.startswith("[") and value.endswith("]")):
            try:
                parsed = json.loads(value)
                return flatten_recursive(parsed)
            except:
                pass # Không phải JSON valid, dùng string gốc
    
    # Giá trị nguyên thủy (str, int, float, bool)
    return value

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

def get_column_mappings(table_name: str) -> dict:
    """Đọc column_mappings từ collections_config"""
    import json
    try:
        conn = sqlite3.connect(FAQ_DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT column_mappings FROM collections_config WHERE name=?", (table_name,))
        row = cur.fetchone()
        conn.close()
        
        if row and row[0]:
            return json.loads(row[0])
        return {}
    except Exception as e:
        print(f"  ⚠️ Error reading column mappings: {e}")
        return {}


def get_db_connection():
    return sqlite3.connect(FAQ_DB_PATH)


def build_embed_text(row_dict: dict, table_name: str, mappings: dict = None) -> str:
    """
    Tạo text để embed. Ưu tiên các trường quan trọng.
    Sử dụng mappings để dịch tên cột sang tiếng Việt.
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
            # Display key: mapped name or original key
            key_display = mappings.get(col, col) if mappings else col
            parts.append(f"{key_display}: {value}")

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


def delete_entire_table_from_qdrant(client: QdrantClient, table_name: str):
    """
    Xóa toàn bộ dữ liệu của một bảng khỏi Qdrant (dùng cho bảng bị exclude)
    """
    print(f"\n[CLEANUP] Checking excluded table: {table_name}")
    existing_ids = get_existing_ids_in_qdrant(client, table_name)
    
    if not existing_ids:
        print("  ✔ No data found in Qdrant. Clean.")
        return

    print(f"  🗑️ Found {len(existing_ids)} items. Deleting...")
    try:
        client.delete(
            collection_name=GLOBAL_COLLECTION_NAME,
            points_selector=list(existing_ids),
        )
        print("  ✅ Deleted all items.")
    except Exception as e:
        print(f"  ⚠️ Error deleting items: {e}")


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

    # Load mappings
    mappings = get_column_mappings(table_name)
    if mappings:
        print(f"  🗺️  Loaded {len(mappings)} column mappings for embedding.")

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

            # 🔥 FLATTEN NGAY TỪ ĐẦU: dùng chung cho embedding + payload
            flat_row = {}
            for k, v in row_dict.items():
                flat_row[k] = flatten_recursive(v)


            text = build_embed_text(flat_row, table_name, mappings)

            # ✅ Build payload từ dữ liệu đã flatten (loại bỏ notion_id)
            payload = {}
            for k, v in flat_row.items():
                if k != "notion_id":
                    payload[k] = v

            payload["source_table"] = table_name
            # 🔥 NEW: Gắn mô tả bảng vào Qdrant
            description = get_table_description_from_sqlite(table_name)
            payload["table_description"] = description
            ids.append(notion_id_str)
            texts_to_embed.append(text)
            payloads.append(payload)

        # DEBUG: Print the first text to prove mapping works
        if texts_to_embed and total_synced == 0:
            print(f"  👀 [DEBUG] First text being embedded:\n     \"{texts_to_embed[0]}\"\n")

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


def create_dynamic_indexes(client: QdrantClient):
    """
    Tự động tạo index cho các cột được dùng làm dynamic filter (cấu hình trong SQLite).
    """
    import json
    from qdrant_client.models import PayloadSchemaType

    print("Checking dynamic indexes...")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT dynamic_filters FROM collections_config WHERE enabled=1")
        rows = cur.fetchall()
        conn.close()

        for row in rows:
            raw = row[0]
            if not raw:
                continue
            try:
                config = json.loads(raw)
                target_col = config.get("target_col")
                if target_col:
                    print(f"  Creating index for dynamic filter column: {target_col}")
                    # Mặc định int cho ID, có thể mở rộng logic check type nếu cần
                    client.create_payload_index(
                        collection_name=GLOBAL_COLLECTION_NAME,
                        field_name=target_col,
                        field_schema=PayloadSchemaType.INTEGER,
                    )
            except Exception as e:
                print(f"  Warning processing filters: {e}")
                
    except Exception as e:
        print(f"Error creating dynamic indexes: {e}")



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
    create_dynamic_indexes(client)

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
        # Filter excluded tables
        filtered_tables = [t for t in all_tables if t not in EXCLUDED_TABLES]
        
        print(f"📋 Found {len(filtered_tables)} tables to sync (Excluded: {len(EXCLUDED_TABLES)}).")
        
        # 1. Sync valid tables
        for table in filtered_tables:
            sync_table_to_global_collection(table, embed_model, client)
            gc.collect()

        # 2. Cleanup excluded tables
        print("\n🧹 Cleaning up excluded tables from Qdrant...")
        for table in EXCLUDED_TABLES:
            delete_entire_table_from_qdrant(client, table)
            gc.collect()

    print("\n🎉 INCREMENTAL SYNC COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()