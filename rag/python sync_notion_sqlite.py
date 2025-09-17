import sqlite3, os
from notion_client import Client
from dotenv import load_dotenv

load_dotenv("rag/.env")
client = Client(auth=os.getenv("NOTION_TOKEN"))
DB_ID = os.getenv("NOTION_DATABASE_ID")

conn = sqlite3.connect("faq.db")
cur = conn.cursor()

cur.execute("""CREATE TABLE IF NOT EXISTS faq (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id TEXT UNIQUE,
    question TEXT,
    answer TEXT,
    updated_at TEXT
)""")

# --- 1. Lấy từ Notion về DB ---
def sync_from_notion():
    resp = client.databases.query(database_id=DB_ID)
    for page in resp["results"]:
        pid = page["id"]
        props = page["properties"]
        q = props["Question"]["rich_text"][0]["plain_text"] if props["Question"]["rich_text"] else ""
        a = props["Answer"]["rich_text"][0]["plain_text"] if props["Answer"]["rich_text"] else ""
        last_edit = page["last_edited_time"]

        cur.execute("SELECT updated_at FROM faq WHERE external_id=?", (pid,))
        row = cur.fetchone()
        if not row:
            cur.execute("INSERT INTO faq(external_id,question,answer,updated_at) VALUES(?,?,?,?)",
                        (pid, q, a, last_edit))
        elif row[0] < last_edit:
            cur.execute("UPDATE faq SET question=?, answer=?, updated_at=? WHERE external_id=?",
                        (q, a, last_edit, pid))
    conn.commit()

# --- 2. Đẩy DB → Notion ---
def sync_to_notion():
    cur.execute("SELECT id, external_id, question, answer, updated_at FROM faq")
    for rid, pid, q, a, upd in cur.fetchall():
        if not pid:  # chưa có trên Notion
            page = client.pages.create(
                parent={"database_id": DB_ID},
                properties={
                    "Question": {"rich_text": [{"text": {"content": q}}]},
                    "Answer": {"rich_text": [{"text": {"content": a}}]}
                }
            )
            cur.execute("UPDATE faq SET external_id=?, updated_at=? WHERE id=?",
                        (page["id"], page["last_edited_time"], rid))
        else:
            # TODO: so sánh với last_edited_time trong Notion
            # nếu DB mới hơn thì update lên Notion
            pass
    conn.commit()

sync_from_notion()
sync_to_notion()
