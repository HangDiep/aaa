# sync_all.py
import os
import sqlite3
import requests
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
# --------------------------
# Config & ENV
# --------------------------
# Đảm bảo mọi file tài nguyên đều được tìm từ thư mục gốc D:\aaa\
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)  # QUAN TRỌNG: đổi thư mục làm việc về D:\aaa
print(f"[DEBUG] Đã chuyển thư mục làm việc về: {BASE_DIR}")
ENV_PATH = r"D:\aaa\rag\data\.env"
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH, override=True)
else:
    load_dotenv()  # fallback

TOKEN = os.getenv("NOTION_TOKEN") or os.getenv("NOTION_API_KEY") or ""
BASE  = (os.getenv("NOTION_BASE_URL") or "https://api.notion.com/v1").rstrip("/")
NV    = os.getenv("NOTION_VERSION", "2022-06-28")
BOOKS_CHECKBOX = os.getenv("BOOKS_CHECKBOX", "Đồng bộ")

DB_FAQ    = os.getenv("DATABASE_ID_FAQ")     # 2a5db606...
DB_BOOKS  = os.getenv("DATABASE_ID_BOOKS")   # 2a4db606...
DB_MAJORS = os.getenv("DATABASE_ID_MAJORS")  # 2a4db606...

DB_PATH = os.path.join(BASE_DIR, "faq.db")

# --------------------------
# HTTP helpers
# --------------------------
def _headers():
    return {
        "Authorization": f"Bearer {TOKEN}",
        "Notion-Version": NV,
        "Content-Type": "application/json",
    }

def _query(dbid, body):
    """Query toàn bộ database Notion với phân trang."""
    url = f"{BASE}/databases/{dbid}/query"
    rs, payload = [], (body or {})
    while True:
        r = requests.post(url, headers=_headers(), json=payload, timeout=30)
        if r.status_code == 401:
            print("[sync] 401 Unauthorized – kiểm tra NOTION_API_KEY và quyền share DB cho integration.")
            return []
        r.raise_for_status()
        j = r.json()
        rs += j.get("results", [])
        if not j.get("has_more") or not j.get("next_cursor"):
            break
        payload = {**(body or {}), "start_cursor": j["next_cursor"]}
    return rs

# --------------------------
# Property extract helpers
# --------------------------
def _txt(blocks):
    """Lấy text từ mảng title/rich_text (kiểu cũ)."""
    if not isinstance(blocks, list) or not blocks:
        return ""
    b0 = blocks[0]
    return (b0.get("plain_text") or (b0.get("text") or {}).get("content") or "").strip()

def _p_txt(prop: dict) -> str:
    """Text từ nhiều kiểu property."""
    if not prop:
        return ""
    if prop.get("title"):
        return "".join([(x.get("plain_text") or (x.get("text") or {}).get("content","") or "") for x in prop["title"]]).strip()
    if prop.get("rich_text"):
        return "".join([(x.get("plain_text") or (x.get("text") or {}).get("content","") or "") for x in prop["rich_text"]]).strip()
    if prop.get("select") and isinstance(prop.get("select"), dict):
        return (prop["select"].get("name") or "").strip()
    if prop.get("multi_select"):
        return ", ".join([(o.get("name") or "").strip() for o in prop["multi_select"] if o.get("name")])
    if prop.get("people"):
        names = []
        for p in prop["people"]:
            nm = (p.get("name") or p.get("email") or "").strip()
            if nm:
                names.append(nm)
        return ", ".join(names)
    if prop.get("rollup"):
        r = prop["rollup"]
        if r.get("type") == "array":
            vals = []
            for x in r.get("array", []):
                if x.get("type") == "title":
                    vals.append(_p_txt({"title": x["title"]}))
                elif x.get("type") == "rich_text":
                    vals.append(_p_txt({"rich_text": x["rich_text"]}))
            return ", ".join([v for v in vals if v])
        if r.get("type") == "number" and r.get("number") is not None:
            return str(r["number"])
    if prop.get("formula"):
        f = prop["formula"]
        if f.get("type") == "string":
            return (f.get("string") or "").strip()
        if f.get("type") == "number" and f.get("number") is not None:
            return str(f["number"])
    return ""

def _p_num(prop: dict):
    """Trả về số từ number | rollup/ formula number | hoặc parse text."""
    if not prop:
        return None
    if prop.get("type") == "number" and prop.get("number") is not None:
        return prop["number"]
    if prop.get("rollup") and prop["rollup"].get("type") == "number":
        return prop["rollup"]["number"]
    if prop.get("formula") and prop["formula"].get("type") == "number":
        return prop["formula"]["number"]
    s = _p_txt(prop)
    try:
        return int(float(s))
    except:
        return None

def _get_prop(props: dict, key: str) -> dict:
    """Tìm property theo key đã chuẩn hóa (bỏ khoảng trắng dư, lower)."""
    key_norm = (key or "").strip().lower()
    for k, v in (props or {}).items():
        if (k or "").strip().lower() == key_norm:
            return v or {}
    return {}

# --------------------------
# DB schema
# --------------------------
def _ensure_tables():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True) if os.path.dirname(DB_PATH) else None
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Tạo bảng
    c.execute("""
        CREATE TABLE IF NOT EXISTS faq(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            notion_id TEXT UNIQUE,
            question TEXT,
            answer TEXT,
            category TEXT,
            language TEXT,
            approved INTEGER,
            last_updated TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS books(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            notion_id TEXT UNIQUE,
            name TEXT,
            author TEXT,
            year INTEGER,
            quantity INTEGER,
            status TEXT,
            major_id TEXT,        -- majors.major_id (mã ngành hiển thị)
            last_updated TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS majors(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            notion_id TEXT UNIQUE,
            major_id TEXT,        -- mã ngành (number/text) do bạn định nghĩa ở Notion
            name TEXT,
            description TEXT
        )
    """)

   

    conn.commit()
    conn.close()

# --------------------------
# Sync FAQ
# --------------------------
def sync_faq():
    if not DB_FAQ:
        print("[sync] FAQ: thiếu DATABASE_ID_FAQ trong .env")
        return

    rows = _query(DB_FAQ, {
        "filter": {"and":[
            {"property":"Question","rich_text":{"is_not_empty":True}},
            {"property":"Answer","rich_text":{"is_not_empty":True}},
            {"property":"Approved","checkbox":{"equals":True}}
        ]},
        "page_size":100
    })

    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    for r in rows:
        p = r.get("properties", {}) or {}
        rid = r.get("id","")  # Notion page id

        qprop = _get_prop(p, "Question")
        question = _p_txt(qprop)

        answer   = _p_txt(_get_prop(p, "Answer"))
        category = _p_txt(_get_prop(p, "Category"))

        # Language có thể là select hoặc rich_text
        lang = "Tiếng Việt"
        lp = _get_prop(p, "Language")
        if lp.get("type") == "select" and lp.get("select"):
            lang = lp["select"].get("name", lang) or lang
        elif lp.get("type") == "rich_text":
            lang = _p_txt(lp) or lang

        last = datetime.now().date().isoformat()
        for name in ["Last Updated","Last Update","Updated","Cập nhật"]:
            dp = _get_prop(p, name).get("date")
            if dp and dp.get("start"):
                last = dp["start"]; break

        if rid and question and answer:
            # UPSERT theo notion_id
            c.execute("""
                INSERT INTO faq(notion_id, question, answer, category, language, approved, last_updated)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(notion_id) DO UPDATE SET
                    question=excluded.question,
                    answer=excluded.answer,
                    category=excluded.category,
                    language=excluded.language,
                    approved=excluded.approved,
                    last_updated=excluded.last_updated
            """, (rid, question, answer, category, lang, 1, last))
    notion_ids = [r.get("id") for r in rows if r.get("id")]
    if notion_ids:
        placeholders = ",".join(["?"] * len(notion_ids))
        c.execute(f"""
            DELETE FROM faq
            WHERE notion_id NOT IN ({placeholders})
        """, notion_ids)
    else:
        print("[sync] faq: Notion trả về 0 dòng – bỏ qua bước xoá.")
    conn.commit(); conn.close()
    print(f"[sync] FAQ: {len(rows)} rows")

# --------------------------
# Sync Majors
# --------------------------
def sync_majors():
    if not DB_MAJORS:
        print("[sync] MAJORS: thiếu DATABASE_ID_MAJORS trong .env")
        return

    rows = _query(DB_MAJORS, {"page_size":100})
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    for r in rows:
        p = r.get("properties", {}) or {}
        rid = r.get("id","")  # Notion page id

        name_prop = _get_prop(p, "Tên ngành")
        name = _p_txt(name_prop)  # title

        # ID ngành có thể là number hoặc rich_text
        mid = ""
        ip = _get_prop(p, "ID ngành")
        if ip.get("type") == "number" and ip.get("number") is not None:
            mid = str(ip["number"])
        elif ip.get("rich_text"):
            mid = _p_txt(ip)

        desc = _p_txt(_get_prop(p, "Mô tả"))

        if rid and name:
            # UPSERT theo notion_id
            c.execute("""
                INSERT INTO majors(notion_id, major_id, name, description)
                VALUES (?,?,?,?)
                ON CONFLICT(notion_id) DO UPDATE SET
                    major_id=excluded.major_id,
                    name=excluded.name,
                    description=excluded.description
            """, (rid, mid, name, desc))
    notion_ids = [r.get("id") for r in rows if r.get("id")]
    if notion_ids:
        placeholders = ",".join(["?"] * len(notion_ids))
        c.execute(f"""
            DELETE FROM majors
            WHERE notion_id NOT IN ({placeholders})
        """, notion_ids)
    else:
        print("[sync] majors: Notion trả về 0 dòng – bỏ qua bước xoá.")


    conn.commit(); conn.close()
    print(f"[sync] MAJORS: {len(rows)} rows")

# --------------------------
# Sync Books (upsert theo Notion notion_id)
# --------------------------
def sync_books():
    if not DB_BOOKS:
        print("[sync] BOOKS: thiếu DATABASE_ID_BOOKS trong .env")
        return

    print("[sync] BOOKS: syncing...")
    rows = _query(DB_BOOKS, {
    "filter": {
        "and": [
            {"property": "Tên sách", "title": {"is_not_empty": True}},
            {"property": "Approved", "checkbox": {"equals": True}}
        ]
    },
    "page_size": 100
})


    total_notions = len(rows)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    insert_count = 0
    update_count = 0
    for r in rows:
        rid = r.get("id", "")                 # Notion page id
        p = r.get("properties", {}) or {}

        name = _p_txt(_get_prop(p, "Tên sách"))
        author = _p_txt(_get_prop(p, "Tác giả"))
        year = _p_num(_get_prop(p, "Năm xuất bản"))
        quantity = _p_num(_get_prop(p, "Số lượng tồn kho"))
        status = _p_txt(_get_prop(p, "Trạng thái"))

        # Relation "Ngành" (Notion page id của ngành)
        rel_prop = _get_prop(p, "Ngành")
        rel = rel_prop.get("relation", []) if isinstance(rel_prop, dict) else []
        major_notion_id = rel[0].get("id") if rel else None

        # Map qua majors.major_id bằng majors.notion_id
        major_id = None
        if major_notion_id:
            c.execute("SELECT major_id FROM majors WHERE notion_id = ?", (major_notion_id,))
            row_major = c.fetchone()
            if row_major:
                major_id = row_major[0]

        last_updated = datetime.now().date().isoformat()

        # Validate tối thiểu
        if not (rid and name):
            continue

        # Kiểm tra tồn tại theo notion_id
        c.execute("SELECT 1 FROM books WHERE notion_id = ?", (rid,))
        exists = c.fetchone() is not None

        # UPSERT theo notion_id
        c.execute("""
            INSERT INTO books(notion_id, name, author, year, quantity, status, major_id, last_updated)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(notion_id) DO UPDATE SET
                name=excluded.name,
                author=excluded.author,
                year=excluded.year,
                quantity=excluded.quantity,
                status=excluded.status,
                major_id=excluded.major_id,
                last_updated=excluded.last_updated
        """, (rid, name, author, year, quantity, status, major_id, last_updated))

        if exists:
            update_count += 1
        else:
            insert_count += 1
       # --- XÓA NHỮNG DÒNG KHÔNG CÒN TRÊN NOTION ---
    notion_ids = [r.get("id") for r in rows if r.get("id")]
    if notion_ids:
        placeholders = ",".join(["?"] * len(notion_ids))
        c.execute(f"DELETE FROM books WHERE notion_id NOT IN ({placeholders})", notion_ids)
    else:
        print("[sync] BOOKS: Notion (Approved filter) trả 0 dòng – skip xoá để an toàn.")


    conn.commit(); conn.close()

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    if not TOKEN:
        print("❌ Thiếu NOTION_API_KEY/NOTION_TOKEN"); raise SystemExit(1)
    _ensure_tables()
    # Chạy majors trước để có mapping ngành
    sync_majors()
    sync_faq()
    sync_books()
    print("[sync] DONE")
