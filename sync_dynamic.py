# ==========================================
# ĐỒ ÁN: Chatbot Dynamic Router - TTN University
# NGÀY NỘP: 21/12/2025
# Copyright © 2025. All rights reserved.
# ==========================================

"""
Dynamic Notion → SQLite → Qdrant Sync
Tự động tạo bảng mới khi phát hiện database mới từ Notion
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
import sqlite3
import os
import json
import subprocess
import requests
import asyncio
from dotenv import load_dotenv

# Get absolute path to database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "faq.db")
print(f"[DYNAMIC SYNC] Using database: {DB_PATH}")
# Tạo router
router = APIRouter(prefix="/notion/dynamic", tags=["notion-dynamic-sync"])

class DynamicSyncPayload(BaseModel):
    """
    Generic payload cho bất kỳ bảng nào
    """
    notion_id: str
    table_name: str  # Tên bảng (VD: thu_vien, faq, books...)
    data: Dict[str, Any]  # Dữ liệu động từ Notion
    approved: Optional[bool] = True
class DeletePayload(BaseModel):
    notion_id: str
    table_name: str
# ==========================
#  DB helper
# ==========================
def get_conn():
    return sqlite3.connect(DB_PATH)
def init_collections_config_table():
    """
    Tạo bảng collections_config để lưu metadata của các collections
    """
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS collections_config (
            name TEXT PRIMARY KEY,
            description TEXT,
            enabled INTEGER DEFAULT 1,
            priority INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT,
            column_mappings TEXT
        )
        """
    )
    # Simple migration for existing table
    try:
        cur.execute("ALTER TABLE collections_config ADD COLUMN column_mappings TEXT")
        print("  ℹ️ Added column_mappings to collections_config")
    except Exception:
        pass # Column likely exists

    # Migration: Add dynamic_filters column
    try:
        cur.execute("ALTER TABLE collections_config ADD COLUMN dynamic_filters TEXT")
        print("  ℹ️ Added dynamic_filters to collections_config")
    except Exception:
        pass # Column likely exists

    conn.commit()
    conn.close()
    print("✅ collections_config table initialized")
def seed_default_filters():
    conn = get_conn()
    cur = conn.cursor()
    
    # Config mặc định cho bảng Sách (sch_)
    sch_filters = {
        "target_col": "id_ngnh",
        "lookup_table": "ngnh",
        "lookup_col_name": "tn_ngnh",
        "lookup_col_id": "id_ngnh",
        "extraction_prompt": "Tìm TÊN NGÀNH trong câu hỏi. Trả về 'null' nếu không có."
    }
    try:
        import json
        filters_json = json.dumps({"id_ngnh": sch_filters}, ensure_ascii=False)
        
        # Danh sách các bảng có khả năng là bảng Sách
        book_tables = ['sch_', 'tra_cu_thng_tin_sch_', 'sach', 'books']
        
        for tbl in book_tables:
            # Kiểm tra xem bảng có tồn tại trong config không
            cur.execute("SELECT name, dynamic_filters FROM collections_config WHERE name=?", (tbl,))
            row = cur.fetchone()
            
            if row:
                if not row[1] or row[1] == "{}": # Nếu chưa có filter hoặc filter rỗng
                    print(f"  🌱 Seeding default filters for '{tbl}'...")
                    sch_json = json.dumps({"id_ngnh": sch_filters}, ensure_ascii=False)
                    cur.execute("UPDATE collections_config SET dynamic_filters = ? WHERE name=?", (sch_json, tbl))
                    conn.commit()
            
    except Exception as e:
        print(f"  ⚠️ Error seeding filters: {e}")
    finally:
        conn.close()

# Initialize table logic
init_collections_config_table()
seed_default_filters()
def sanitize_table_name(name: str) -> str:
    """Làm sạch tên bảng (chỉ cho phép a-z, 0-9, _)"""
    import re

    clean = name.lower().replace(" ", "_").replace("-", "_")
    clean = re.sub(r"[^a-z0-9_]", "", clean)
    return clean
def sanitize_column_name(name: str) -> str:
    """Làm sạch tên cột"""
    import re

    clean = name.lower().replace(" ", "_").replace("-", "_")
    clean = re.sub(r"[^a-z0-9_]", "", clean)
    return clean
def infer_sql_type(value: Any) -> str:
    """Tự động phát hiện kiểu dữ liệu SQL"""
    if value is None:
        return "TEXT"
    elif isinstance(value, bool):
        return "INTEGER"
    elif isinstance(value, int):
        return "INTEGER"
    elif isinstance(value, float):
        return "REAL"
    else:
        return "TEXT"
def sanitize_sql_value(value):
    """
    Chuyển đổi list/dict thành JSON string để lưu vào SQLite
    Tránh lỗi 'type list is not supported'
    """
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value
def generate_table_description(table_name: str, data: Dict[str, Any]) -> str:
    """
    Tự động tạo mô tả cho bảng mới bằng LLM
    """
    from dotenv import load_dotenv
    import requests

    # Load env
    # Load env
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ENV_PATH = os.path.join(BASE_DIR, "rag", ".env")
    try:
        if os.path.exists(ENV_PATH):
            load_dotenv(ENV_PATH, override=True)
    except Exception:
        pass

    ZIPUR_API_KEY = os.getenv("ZIPUR_API_KEY")
    ZIPUR_MODEL = os.getenv("ZIPUR_MODEL", "glm-4-plus")

    if not ZIPUR_API_KEY:
        columns = [sanitize_column_name(k) for k in data.keys()]
        return f"Bảng {table_name} chứa: {', '.join(columns[:5])}"
    columns = list(data.keys())

    sample_values = []
    for k, v in data.items():
        if v and len(str(v)) < 50:
            sample_values.append(f"{k}: {v}")
    sample_str = "; ".join(sample_values[:5])

    prompt = f"""Bảng "{table_name}" chứa dữ liệu mẫu: [{sample_str}]

Dựa vào tên bảng và dữ liệu mẫu trên, hãy viết 1 câu mô tả ngắn gọn (30-50 từ) về mục đích của bảng này.

Ví dụ:
- Bảng "books" (name: Python Basics; author: John Doe) → "Chứa thông tin các đầu sách, tài liệu và tác giả."
- Bảng "store" (item: Cafe; price: 20k) → "Thông tin thực đơn, bảng giá đồ uống tại cửa hàng."

Chỉ viết mô tả, không thêm gì khác:"""

    try:
        headers = {
            "Authorization": f"Bearer {ZIPUR_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": ZIPUR_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 50,
        }

        resp = requests.post(
            "https://open.bigmodel.cn/api/paas/v4/chat/completions",
            headers=headers,
            json=payload,
            timeout=10,
        )

        if resp.status_code == 200:
            data_resp = resp.json()
            description = data_resp["choices"][0]["message"]["content"].strip()
            return description
        else:
            columns_str = ", ".join(columns[:5])
            return f"Bảng {table_name} chứa: {columns_str}"

    except Exception as e:
        print(f"⚠️ Lỗi generate description: {e}")
        columns_str = ", ".join(columns[:5])
        return f"Bảng {table_name} chứa: {columns_str}"
def save_to_collections_config(table_name: str, description: str, mappings: dict = None):
    """
    Lưu thông tin collection vào collections_config
    """
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    
    mappings_json = json.dumps(mappings, ensure_ascii=False) if mappings else "{}"

    cur.execute(
        """
        INSERT INTO collections_config 
        (name, description, enabled, priority, created_at, updated_at, column_mappings)
        VALUES (?, ?, 1, 0, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            description = excluded.description,
            updated_at = excluded.updated_at,
            column_mappings = excluded.column_mappings
        """,
        (table_name, description, now, now, mappings_json),
    )

    conn.commit()
    conn.close()
    print(f"  💾 Saved to collections_config: {table_name}")
def update_collection_mappings(table_name: str, mappings: dict):
    """
    Chỉ update column_mappings cho bảng (dùng cho bảng đã tồn tại)
    """
    if not mappings:
        return

    conn = get_conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    mappings_json = json.dumps(mappings, ensure_ascii=False)

    # Insert ignore to ensure row exists, then update
    # But simpler: assume row exists or we don't care about description here.
    # actually scan_new_databases ensures table structure.
    
    cur.execute(
        """
        UPDATE collections_config
        SET column_mappings = ?, updated_at = ?
        WHERE name = ?
        """,
        (mappings_json, now, table_name)
    )
    
    # If no row updated (should not happen if created), we could Insert, but description would be missing.
    # rely on create_table_if_not_exists to handle creation.
    
    conn.commit()
    conn.close()
# Tự tạo bảng SQLite  mới
def create_table_if_not_exists(table_name: str, data: Dict[str, Any]):
    """
    Tự động tạo bảng SQLite nếu chưa tồn tại.
    Nếu bảng đã tồn tại → Kiểm tra và ADD COLUMN mới nếu thiếu.
    """
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    )
    exists = cur.fetchone()

    expected_columns = {
        "notion_id": "TEXT PRIMARY KEY",
        "last_updated": "TEXT",
        "approved": "INTEGER DEFAULT 1",
    }
    for key, value in data.items():
        col_name = sanitize_column_name(key)
        if col_name not in expected_columns:
            sql_type = infer_sql_type(value)#đoán dữ liệu
            expected_columns[col_name] = sql_type

    if not exists:
        print(f"  🆕 Creating new table: {table_name}")

        cols_sql = []
        for col, dtype in expected_columns.items():
            cols_sql.append(f"{col} {dtype}")

        create_sql = f"CREATE TABLE {table_name} ({', '.join(cols_sql)})"
        cur.execute(create_sql)
        conn.commit()
        print(f"  ✅ Table '{table_name}' created successfully!")

        print(f"  🤖 Generating description for '{table_name}'...")
        description = generate_table_description(table_name, data) 
        #upsert_dynamic_data
        # Capture mappings: {slug: original_name}
        mappings = {sanitize_column_name(k): k for k in data.keys()}
        save_to_collections_config(table_name, description, mappings)
        #async def dynamic_delete(payload: DeletePayload):

    else:
        cur.execute(f"PRAGMA table_info({table_name})")
        existing_cols = {row[1] for row in cur.fetchall()}

        missing_cols = []
        for col_name, dtype in expected_columns.items():
            if col_name not in existing_cols:
                missing_cols.append((col_name, dtype))

        if missing_cols:
            print(f"  ⚠️ Schema Changed inside '{table_name}'. Migration needed.")
            for col_name, dtype in missing_cols:
                try:
                    alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {col_name} {dtype}"
                    print(f"     ➕ Adding column: {col_name} ({dtype})")
                    cur.execute(alter_sql)
                except Exception as e:
                    print(f"     ❌ Failed to add column {col_name}: {e}")
            conn.commit()
            print("  ✅ Schema migration completed.")

    conn.close()
def upsert_dynamic_data(
    table_name: str, notion_id: str, data: Dict[str, Any], approved: bool = True
):
    """
    Insert hoặc Update dữ liệu vào bảng động (AN TOÀN TUYỆT ĐỐI với list/dict)
    """
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()

    columns = ["notion_id"]
    values = [sanitize_sql_value(notion_id)]

    for key, value in data.items():
        col_name = sanitize_column_name(key)
        if col_name == "notion_id":
            continue

        safe_value = sanitize_sql_value(value)

        if isinstance(safe_value, (list, dict)):
            print(
                f"❌ [SQLITE BLOCKED] Column: {col_name} | Value: {safe_value} | Type: {type(safe_value)}"
            )
            safe_value = json.dumps(safe_value, ensure_ascii=False)

        columns.append(col_name)
        values.append(
            str(safe_value) if isinstance(safe_value, (list, dict)) else safe_value
        )

    columns.extend(["last_updated", "approved"])
    values.extend([now, 1 if approved else 0])

    placeholders = ", ".join(["?"] * len(columns))
    update_clause_items = []
    for col in columns:
        if col != "notion_id":
            update_clause_items.append(f"{col} = excluded.{col}")
    update_clause = ", ".join(update_clause_items)

    sql = f"""
        INSERT INTO {table_name} ({', '.join(columns)})
        VALUES ({placeholders})
        ON CONFLICT(notion_id) DO UPDATE SET {update_clause}
    """

    try:
        cur.execute(sql, values)
        conn.commit()
    finally:
        conn.close()
def delete_dynamic_data(table_name: str, notion_id: str):
    """Xóa dữ liệu khỏi bảng động"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"DELETE FROM {table_name} WHERE notion_id = ?", (notion_id,))
    conn.commit()
    conn.close()
# ==========================
#  Notion Parser Helper
# ==========================
def parse_notion_properties(props: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested Notion properties into simple key-value pairs.
    Handles: Title, RichText, Select, MultiSelect, Date, Number, Checkbox, etc.
    """
    data = {}
    for key, prop in props.items():
        if not isinstance(prop, dict):
            data[key] = prop
            continue

        prop_type = prop.get("type")
        value = None

        try:
            if prop_type == "title":
                value = "".join(
                    [t.get("plain_text", "") for t in prop.get("title", [])]
                )
            elif prop_type == "rich_text":
                value = "".join(
                    [t.get("plain_text", "") for t in prop.get("rich_text", [])]
                )
            elif prop_type == "select":
                select = prop.get("select")
                value = select.get("name") if select else None
            elif prop_type == "multi_select":
                multi = prop.get("multi_select", [])
                value = ", ".join([opt.get("name", "") for opt in multi])
            elif prop_type == "date":
                date = prop.get("date")
                value = date.get("start") if date else None
            elif prop_type == "checkbox":
                value = prop.get("checkbox")
            elif prop_type == "number":
                value = prop.get("number")
            elif prop_type == "email":
                value = prop.get("email")
            elif prop_type == "phone_number":
                value = prop.get("phone_number")
            elif prop_type == "url":
                value = prop.get("url")
            elif prop_type == "status":
                status = prop.get("status")
                value = status.get("name") if status else None
            else:
                if prop_type in prop:
                    value = prop[prop_type]
                    if isinstance(value, dict) and "name" in value:
                        value = value["name"]
        except Exception as e:
            print(f"⚠️ Error parsing property '{key}': {e}")
            value = str(prop)

        data[key] = sanitize_sql_value(value)

    return data
# ==========================
#  API Endpoints
# ==========================
@router.post("/sync")
async def dynamic_sync(request: Request):
    """
    Smart Endpoint: ONLY accept table_name from BODY (no query param anymore)
    """
    try:
        payload = await request.json()

        target_table = payload.get("table_name")
        if not target_table:
            raise HTTPException(
                status_code=422,
                detail="Missing 'table_name' in request body",
            )

        target_table = sanitize_table_name(target_table)

        notion_id = payload.get("notion_id") or payload.get("id")
        if not notion_id:
            raise HTTPException(
                status_code=422,
                detail="Missing 'notion_id' or 'id' in body.",
            )

        # SMART PARSER
        if "data" in payload and isinstance(payload["data"], dict):
            final_data = payload["data"]
            approved = payload.get("approved", True)
        elif "properties" in payload:
            print(
                f"🧠 [SMART PARSER] Detected RAW Notion Payload for '{target_table}'"
            )
            final_data = parse_notion_properties(payload["properties"])

            approved = True
            if "Approved" in final_data:
                val = final_data["Approved"]
                if val is not None:
                    approved = bool(val)
        else:
            print("⚠️ [SMART PARSER] Unknown structure. Using flat payload.")
            final_data = {
                k: v
                for k, v in payload.items()
                if k not in ["notion_id", "id", "table_name"]
            }
            approved = True

        # Skip nếu chưa Approved (trừ bảng 'nganh')
        if target_table != "nganh" and not approved:
            print(
                f"⏭️ [DYNAMIC SYNC] Unapproved => delete row in '{target_table}' | ID: {notion_id}"
            )
            try:
                delete_dynamic_data(target_table, notion_id)
            except Exception as e:
                print(f"⚠️ Error deleting unapproved row: {e}")

            return {
                "status": "deleted_unapproved",
                "table_name": target_table,
                "notion_id": notion_id,
                "reason": "not_approved",
            }

        # Sanitize toàn bộ value
        safe_data = {}
        for k, v in final_data.items():
            if isinstance(v, (list, dict)):
                safe_data[k] = json.dumps(v, ensure_ascii=False)
            else:
                safe_data[k] = v
        final_data = safe_data

        print(f"📥 [DYNAMIC SYNC] Table: '{target_table}' | ID: {notion_id}")

        create_table_if_not_exists(target_table, final_data)
        upsert_dynamic_data(target_table, notion_id, final_data, approved)

        print(f"✅ [DYNAMIC SYNC] Success: '{target_table}'")

        try:
            subprocess.Popen(
                ["python", "push_to_qdrant_dynamic.py", target_table],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"⚠️ Qdrant trigger error: {e}")

        try:
            import requests

            requests.post("http://localhost:8000/reload-config", timeout=2)
        except Exception:
            pass

        return {
            "status": "ok",
            "table_name": target_table,
            "notion_id": notion_id,
            "parsed_fields": list(final_data.keys()),
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [DYNAMIC SYNC] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/delete")
@router.delete("/delete")
async def dynamic_delete(payload: DeletePayload):
    """
    Xóa record khỏi bảng động
    """
    try:
        table_name = sanitize_table_name(payload.table_name)

        print(f"🗑️  [DYNAMIC DELETE] Deleting from '{table_name}'")
        print(f"   Notion ID: {payload.notion_id}")

        delete_dynamic_data(table_name, payload.notion_id)

        print(f"✅ [DYNAMIC DELETE] Deleted from '{table_name}'")

        return {
            "status": "deleted",
            "table_name": table_name,
            "notion_id": payload.notion_id,
        }

    except Exception as e:
        print(f"❌ [DYNAMIC DELETE] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/tables")
async def list_tables():
    """
    Liệt kê tất cả các bảng trong SQLite
    """
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cur.fetchall()]
        conn.close()

        return {"status": "ok", "tables": tables, "count": len(tables)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/schema/{table_name}")
async def get_table_schema(table_name: str):
    """
    Xem schema của một bảng
    """
    try:
        table_name = sanitize_table_name(table_name)

        conn = get_conn()
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table_name})")
        columns = cur.fetchall()
        conn.close()

        if not columns:
            raise HTTPException(
                status_code=404, detail=f"Table '{table_name}' not found"
            )

        schema = [
            {
                "cid": col[0],
                "name": col[1],
                "type": col[2],
                "notnull": bool(col[3]),
                "default": col[4],
                "pk": bool(col[5]),
            }
            for col in columns
        ]

        return {"status": "ok", "table_name": table_name, "schema": schema}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/scan")
async def scan_new_databases():
    """
    AUTO-DISCOVERY: Quét toàn bộ Notion workspace để tìm database mới.
    Điều kiện: Bạn phải "Share" database đó với Integration (Con Bot).
    """
    import requests

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ENV_PATH = os.path.join(BASE_DIR, "rag", ".env")
    try:
        if os.path.exists(ENV_PATH):
            from dotenv import load_dotenv

            load_dotenv(ENV_PATH, override=True)
    except Exception:
        pass

    NOTION_API_KEY = os.getenv("NOTION_API_KEY")
    NOTION_VERSION = "2022-06-28"

    if not NOTION_API_KEY:
        raise HTTPException(status_code=500, detail="Missing NOTION_API_KEY in .env")

    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }

    print("🔍 [AUTO-SCAN] Searching for accessible databases...")
    search_url = "https://api.notion.com/v1/search"
    search_payload = {
        "filter": {"value": "database", "property": "object"},
        "page_size": 100,
    }

    try:
        resp = requests.post(
            search_url, headers=headers, json=search_payload, timeout=15
        )
        if resp.status_code != 200:
            raise Exception(f"Notion Search Error: {resp.text}")

        results = resp.json().get("results", [])
        print(f"✅ Found {len(results)} databases accessible.")

        found_names = []
        for d in results:
            t = "".join([x.get("plain_text", "") for x in d.get("title", [])])
            found_names.append(t if t else d["id"])
        print(f"📋 List: {found_names}")

        synced_tables = []

        for db in results:
            db_id = db["id"]

            raw_title = ""
            if db.get("title"):
                raw_title = "".join(
                    [t.get("plain_text", "") for t in db.get("title", [])]
                )

            if not raw_title:
                raw_title = f"db_{db_id[:4]}"

            table_name = sanitize_table_name(raw_title)
            print(f"   👉 Checking DB: {raw_title} -> Table: {table_name}")

            query_url = f"https://api.notion.com/v1/databases/{db_id}/query"

            all_rows = []
            has_more = True
            next_cursor = None
            page_count = 0

            print(f"      🔄 Syncing data from '{table_name}'...")

            while has_more:
                payload = {"page_size": 100}
                if next_cursor:
                    payload["start_cursor"] = next_cursor

                try:
                    q_resp = requests.post(
                        query_url, headers=headers, json=payload, timeout=30
                    )
                    if q_resp.status_code != 200:
                        print(f"      ⚠️ Error fetching page: {q_resp.text}")
                        break

                    data = q_resp.json()
                    rows = data.get("results", [])
                    all_rows.extend(rows)

                    has_more = data.get("has_more", False)
                    next_cursor = data.get("next_cursor")
                    page_count += 1

                    print(f"- Page {page_count}: fetched {len(rows)} rows.")

                    if len(all_rows) > 1000:
                        print("      ⚠️ Reached 1000 rows limit per scan. Stopping.")
                        break
                except Exception as e:
                    print(f"      ❌ Network error: {e}")
                    break

            if not all_rows:
                print("      ⚠️  Empty Database. Skipping.")
                continue

            for row in all_rows:
                notion_id = row["id"]
                props = row.get("properties", {})

                final_data = parse_notion_properties(props)

                approved = True
                if table_name != "nganh" and "Approved" in final_data:
                    val = final_data["Approved"]
                    if val is not None:
                        approved = bool(val)

                                # Nếu không Approved → xoá khỏi SQLite nếu đã tồn tại rồi, rồi bỏ qua
                if table_name != "nganh" and not approved:
                    print(
                        f"⏭️ [SCAN] Unapproved => delete row | table={table_name} | id={notion_id}"
                    )
                    try:
                        delete_dynamic_data(table_name, notion_id)
                    except Exception as e:
                        print(f"⚠️ Error deleting unapproved row in scan: {e}")
                    continue


                safe_data = {k: sanitize_sql_value(v) for k, v in final_data.items()}

                create_table_if_not_exists(table_name, safe_data)
                upsert_dynamic_data(table_name, notion_id, safe_data, approved=approved)

            print(f"      ✅ Synced {len(all_rows)} items.")

            try:
                subprocess.Popen(
                    ["python", "push_to_qdrant_dynamic.py", table_name],# gọi push to qdrant
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

            try:
                # Capture mappings from properties of the first row (or merged)
                # to Ensure existing tables get mappings updated
                if all_rows:
                    sample_props = all_rows[0].get("properties", {})
                    # For better coverage, maybe merge keys from a few rows?
                    # But usually schema is consistent.
                    mappings = {sanitize_column_name(k): k for k in sample_props.keys()}
                    update_collection_mappings(table_name, mappings)
                    print(f"      🗺️  Updated column mappings for '{table_name}'")
            except Exception as e:
                print(f"      ⚠️ Failed to update mappings: {e}")

            synced_tables.append(table_name)

        # AUTO-DELETE stale tables
        if len(found_names) > 0:
            try:
                conn = get_conn()
                cur = conn.cursor()

                cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                verify_rows = cur.fetchall()
                existing_tables = [r[0] for r in verify_rows]

                WHITELIST = {
                    "sqlite_sequence",
                    "conversations",
                    "questions_log",
                    "collections_config",
                    "sync_meta",
                }

                active_set = set(synced_tables)
                stale_tables = []

                for tbl in existing_tables:
                    if tbl not in active_set and tbl not in WHITELIST:
                        stale_tables.append(tbl)

                for bad_table in stale_tables:
                    print(
                        f"   🗑️ [Auto-Delete] Table '{bad_table}' no longer exists in Notion. Dropping..."
                    )
                    cur.execute(f'DROP TABLE IF EXISTS "{bad_table}"')

                conn.commit()
                conn.close()
                
                # NEW: Also cleanup collections_config
                cleanup_deleted_tables_in_sqlite(synced_tables)
                
            except Exception as e:
                print(f"   ⚠️ [Auto-Delete] Error: {e}")

        return {
            "status": "ok",
            "message": "Auto-Discovery Completed",
            "synced_tables": synced_tables,
            "count": len(synced_tables),
            "accessible_databases": found_names,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Scan Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def cleanup_deleted_tables_in_sqlite(valid_tables):
    """
    Xóa các bảng trong collections_config mà không còn tồn tại trên Notion.
    valid_tables: danh sách các bảng hiện có (được sync thành công).
    """
    try:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute("SELECT name FROM collections_config")
        existing_configs = [row[0] for row in cur.fetchall()]

        # WHITELIST: Bảng hệ thống không bao giờ xoá khỏi config (nếu có)
        WHITELIST_CONFIG = {"global", "faq"} 

        to_delete = [t for t in existing_configs if t not in valid_tables and t not in WHITELIST_CONFIG]

        if to_delete:
            print(f"🧹 [Cleanup] Xóa {len(to_delete)} bảng khỏi collections_config:")
            for t in to_delete:
                cur.execute("DELETE FROM collections_config WHERE name = ?", (t,))
                print(f"   - Đã xóa: {t}")
            conn.commit()
        else:
            print("✔ [Cleanup] Không có bảng nào cần xóa khỏi collections_config.")

        conn.close()
    except Exception as e:
        print(f"⚠ [Cleanup] Lỗi khi dọn dẹp collections_config: {e}")
if __name__ == "__main__":
    print("🚀 [MANUAL TRIGGER] Starting Notion Sync...")
    try:
        asyncio.run(scan_new_databases())
        print("✅ [MANUAL TRIGGER] Sync Completed!")
    except Exception as e:
        print(f"❌ [MANUAL TRIGGER] Failed: {e}")
