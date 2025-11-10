# sync_faq.py
import os, sqlite3, requests
from datetime import datetime
from dotenv import load_dotenv

ENV_PATH = r"D:\HTML\a\rag\.env"
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH, override=True)

TOKEN = os.getenv("NOTION_TOKEN") or os.getenv("NOTION_API_KEY") or ""
DBID  = os.getenv("NOTION_DATABASE_ID") or os.getenv("DATABASE_ID_FAQ") or ""
BASE  = (os.getenv("NOTION_BASE_URL") or "https://api.notion.com/v1").rstrip("/")
FAQ_DB_PATH = os.path.join(os.path.dirname(__file__), "faq.db")

def ensure_faq_db():
    os.makedirs(os.path.dirname(FAQ_DB_PATH), exist_ok=True) if os.path.dirname(FAQ_DB_PATH) else None
    conn = sqlite3.connect(FAQ_DB_PATH); cur = conn.cursor()
    cur.execute("""
      CREATE TABLE IF NOT EXISTS faq(
        id TEXT PRIMARY KEY, question TEXT, answer TEXT,
        category TEXT, language TEXT, approved INTEGER, last_updated TEXT)
    """)
    conn.commit(); conn.close()

def _txt(block):
    if not isinstance(block, list) or not block: return ""
    return (block[0].get("plain_text") or block[0].get("text",{}).get("content") or "").strip()

def upsert(rows):
    conn = sqlite3.connect(FAQ_DB_PATH); cur = conn.cursor()
    for row in rows:
        p = row.get("properties", {})
        rid = row.get("id","")
        qprop = p.get("Question", {})
        question = _txt(qprop.get(qprop.get("type",""), []))
        answer   = _txt(p.get("Answer", {}).get("rich_text", []))
        csel     = p.get("Category", {}).get("select")
        category = (csel or {}).get("name","")
        lang = "Tiếng Việt"
        lprop = p.get("Language",{})
        if lprop.get("type") == "select" and lprop.get("select"):
            lang = lprop["select"].get("name", lang)
        elif lprop.get("type") == "rich_text":
            lang = _txt(lprop.get("rich_text", [])) or lang
        approved = 1 if p.get("Approved",{}).get("checkbox", False) else 0
        last_updated = datetime.now().date().isoformat()
        for name in ["Last Updated","Last Update","Updated","Cập nhật"]:
            d = p.get(name,{}).get("date")
            if d and d.get("start"): last_updated = d["start"]; break
        if rid and question and answer:
            cur.execute("""
              INSERT OR REPLACE INTO faq
              (id, question, answer, category, language, approved, last_updated)
              VALUES (?,?,?,?,?,?,?)
            """, (rid, question, answer, category, lang, approved, last_updated))
    conn.commit(); conn.close()

def pull_once():
    if not TOKEN or not DBID:
        print("[sync] thiếu token/dbid"); return
    url = f"{BASE}/databases/{DBID}/query"
    headers = {"Authorization": f"Bearer {TOKEN}",
               "Notion-Version": os.getenv("NOTION_VERSION","2022-06-28"),
               "Content-Type":"application/json"}
    body = {"filter":{"and":[
                {"property":"Question","rich_text":{"is_not_empty":True}},
                {"property":"Answer","rich_text":{"is_not_empty":True}},
                {"property":"Approved","checkbox":{"equals":True}}
            ]}, "page_size":100}
    results=[]; payload=body
    while True:
        r = requests.post(url, headers=headers, json=payload, timeout=20)
        if r.status_code==401: print("[sync] 401 Unauthorized"); return
        r.raise_for_status()
        data = r.json(); results += data.get("results",[])
        nxt = data.get("next_cursor")
        if not data.get("has_more") or not nxt: break
        payload = {**body, "start_cursor": nxt}
    ensure_faq_db(); upsert(results)
    print(f"[sync] OK: {len(results)} rows")

if __name__ == "__main__":
    pull_once()
