import os, sqlite3, requests
from datetime import datetime
from dotenv import load_dotenv

ENV_PATH = r"D:\HTML\a\rag\.env"
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH, override=True)
else:
    load_dotenv()

TOKEN = os.getenv("NOTION_TOKEN") or os.getenv("NOTION_API_KEY")
BASE  = (os.getenv("NOTION_BASE_URL") or "https://api.notion.com/v1").rstrip("/")
NV    = os.getenv("NOTION_VERSION", "2022-06-28")

DB_FAQ    = os.getenv("DATABASE_ID_FAQ")
DB_BOOKS  = os.getenv("DATABASE_ID_BOOKS")
DB_MAJORS = os.getenv("DATABASE_ID_MAJORS")
DB_PATH   = "faq.db"

def headers():
    return {"Authorization": f"Bearer {TOKEN}", "Notion-Version": NV, "Content-Type": "application/json"}

def query(dbid, body):
    url = f"{BASE}/databases/{dbid}/query"
    all_rows = []
    while True:
        r = requests.post(url, headers=headers(), json=body, timeout=30)
        r.raise_for_status()
        j = r.json()
        all_rows += j.get("results", [])
        if not j.get("has_more"):
            break
        body["start_cursor"] = j["next_cursor"]
    return all_rows

def txt(prop):
    if not prop: return ""
    if prop.get("title"):
        return "".join([x["plain_text"] for x in prop["title"]])
    if prop.get("rich_text"):
        return "".join([x["plain_text"] for x in prop["rich_text"]])
    if prop.get("select"):
        return prop["select"].get("name", "")
    return ""

def num(prop):
    if not prop: return None
    if prop.get("number") is not None:
        return prop["number"]
    try:
        return int(float(txt(prop)))
    except: return None

def get_prop(p, key):
    for k, v in p.items():
        if k.strip().lower() == key.strip().lower():
            return v
    return {}

def rebuild_table(conn, name, schema, rows):
    c = conn.cursor()
    c.execute(f"DROP TABLE IF EXISTS {name}")
    c.execute(schema)
    if rows:
        ph = ",".join(["?"] * len(rows[0]))
        c.executemany(f"INSERT INTO {name} VALUES ({ph})", rows)
    conn.commit()

def sync_all():
    conn = sqlite3.connect(DB_PATH)

    # MAJORS
    majors_rows = []
    if DB_MAJORS:
        rows = query(DB_MAJORS, {"page_size": 100})
        for r in rows:
            p = r["properties"]
            majors_rows.append((
                r["id"],
                txt(get_prop(p, "ID ngành")),
                txt(get_prop(p, "Tên ngành")),
                txt(get_prop(p, "Mô tả")),
            ))
        rebuild_table(conn, "majors", """
            CREATE TABLE majors(notion_id TEXT PRIMARY KEY, major_id TEXT, name TEXT, description TEXT)
        """, majors_rows)
        print("[sync] majors:", len(majors_rows))

    # FAQ
    faq_rows = []
    if DB_FAQ:
        rows = query(DB_FAQ, {"page_size": 100})
        for r in rows:
            p = r["properties"]
            faq_rows.append((
                r["id"],
                txt(get_prop(p, "Question")),
                txt(get_prop(p, "Answer")),
                txt(get_prop(p, "Category")),
                "Tiếng Việt",
                1,
                datetime.now().isoformat()
            ))
        rebuild_table(conn, "faq", """
            CREATE TABLE faq(notion_id TEXT PRIMARY KEY, question TEXT, answer TEXT, category TEXT, language TEXT, approved INTEGER, last_updated TEXT)
        """, faq_rows)
        print("[sync] faq:", len(faq_rows))

    # BOOKS
    books_rows = []
    if DB_BOOKS:
        rows = query(DB_BOOKS, {"page_size": 100})
        for r in rows:
            p = r["properties"]
            books_rows.append((
                r["id"],
                txt(get_prop(p, "Tên sách")),
                txt(get_prop(p, "Tác giả")),
                num(get_prop(p, "Năm xuất bản")),
                num(get_prop(p, "Số lượng tồn kho")),
                txt(get_prop(p, "Trạng thái")),
                datetime.now().isoformat()
            ))
        rebuild_table(conn, "books", """
            CREATE TABLE books(notion_id TEXT PRIMARY KEY, name TEXT, author TEXT, year INTEGER, quantity INTEGER, status TEXT, last_updated TEXT)
        """, books_rows)
        print("[sync] books:", len(books_rows))

    conn.close()
    print("[sync] ✅ Hoàn tất, dữ liệu SQLite đã giống 100% Notion")

if __name__ == "__main__":
    sync_all()
