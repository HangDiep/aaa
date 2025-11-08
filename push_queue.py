# push_queue.py
import os, sqlite3, time
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ENV_PATH = r"D:\HTML\a\rag\.env"  # sửa nếu khác
CHAT_DB   = r"D:\HTML\a\chat.db"

def rt(txt: str):
    return [{"type":"text","text":{"content":txt or ""}}]

def session_with_retry():
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.8,
                  status_forcelist=[429,500,502,503,504],
                  allowed_methods=["POST"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

def load_env():
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH, override=True)
    token = os.getenv("NOTION_API_KEY") or os.getenv("NOTION_TOKEN") or ""
    dbid  = os.getenv("DATABASE_ID_FAQ") or os.getenv("NOTION_DATABASE_ID") or ""
    base  = (os.getenv("NOTION_BASE_URL") or "https://ntn-api.notion.so/v1").rstrip("/")
    if not token or not dbid:
        raise SystemExit("❌ Thiếu NOTION_API_KEY hoặc DATABASE_ID_FAQ trong .env")
    return token, dbid, base

def fetch_unsent(limit=50):
    con = sqlite3.connect(CHAT_DB)
    cur = con.cursor()
    # tạo bảng theo dõi nếu chưa có
    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations_sync (
      conv_id INTEGER PRIMARY KEY,
      notion_page_id TEXT,
      synced INTEGER DEFAULT 0,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    # lấy các dòng chưa sync
    rows = cur.execute("""
      SELECT c.id, c.user_message, c.bot_reply
      FROM conversations c
      LEFT JOIN conversations_sync s ON s.conv_id = c.id
      WHERE s.conv_id IS NULL OR s.synced = 0
      ORDER BY c.id ASC
      LIMIT ?
    """, (limit,)).fetchall()
    con.close()
    return rows

def mark_synced(conv_id, page_id=""):
    con = sqlite3.connect(CHAT_DB)
    cur = con.cursor()
    cur.execute("""
      INSERT INTO conversations_sync (conv_id, notion_page_id, synced, created_at)
      VALUES (?, ?, 1, CURRENT_TIMESTAMP)
      ON CONFLICT(conv_id) DO UPDATE SET
        notion_page_id=excluded.notion_page_id, synced=1, created_at=CURRENT_TIMESTAMP
    """, (conv_id, page_id))
    con.commit()
    con.close()

def post_page(sess, base, token, dbid, q, a):
    url = f"{base}/pages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Notion-Version": os.getenv("NOTION_VERSION","2022-06-28"),
        "Host": "ntn-api.notion.so" if "ntn-api.notion.so" in base else "api.notion.com",
    }
    payload = {
        "parent": {"database_id": dbid},
        "properties": {
            "Question": {"rich_text": rt(q)},
            "Answer":   {"rich_text": rt(a)},
            "Approved": {"checkbox": False},
            "Language": {"select": {"name":"Tiếng Việt"}},
        },
    }
    r = sess.post(url, headers=headers, json=payload, timeout=45, allow_redirects=True)
    return r

def main():
    token, dbid, base = load_env()
    sess = session_with_retry()
    rows = fetch_unsent(limit=100)
    if not rows:
        print("✅ Không có dòng cần đẩy.")
        return
    ok = fail = 0
    for conv_id, q, a in rows:
        q = (q or "").strip()
        a = (a or "").strip()
        if not q:
            mark_synced(conv_id, "")
            continue
        try:
            r = post_page(sess, base, token, dbid, q, a)
            if r.status_code in (200, 201):
                mark_synced(conv_id, r.json().get("id",""))
                ok += 1
            else:
                print(f"[Notion HTTP] {r.status_code} {r.text[:200]}")
                # 530/timeout → để lần sau chạy lại
                fail += 1
        except requests.exceptions.Timeout:
            print("[Notion] Timeout – sẽ thử lại sau.")
            fail += 1
        except Exception as e:
            print("[Notion] Error:", e)
            fail += 1
        time.sleep(0.3)
    print(f"✅ Đẩy OK: {ok} | ❌ Lỗi: {fail}")

if __name__ == "__main__":
    main()
