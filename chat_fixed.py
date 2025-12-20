import os, random, json, sqlite3, re, time
# chat_fixed.py
import threading
from dotenv import load_dotenv
from typing import Optional, List, Dict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import socket
from datetime import datetime
import chat
import requests 
from fastapi.responses import PlainTextResponse, HTMLResponse, FileResponse
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from sync_dynamic import router as dynamic_router  # Import dynamic sync router

app = FastAPI()

# Mount static files (CSS, JS)
app.mount("/view", StaticFiles(directory="view"), name="view")

# Include dynamic sync endpoints từ sync_dynamic.py
app.include_router(dynamic_router)
print("✅ Dynamic sync endpoints included: /notion/dynamic/sync, /notion/dynamic/delete")

# ============== RELOAD CONFIG ENDPOINT ==============
@app.get("/")
async def root():
    """Trang chủ Landing Page"""
    return FileResponse("view/index.html")

@app.get("/chatbot")
async def chatbot_page():
    """Giao diện Chatbot (dùng cho iframe)"""
    return FileResponse("view/Chatbot.html")

@app.post("/reload-config")
def reload_config():
    """
    Endpoint để reload collections config sau khi thêm bảng mới
    Gọi endpoint này để chat.py nhận diện bảng mới ngay lập tức
    """
    try:
        from chat_dynamic_router import trigger_config_reload
        collections = trigger_config_reload()
        return {
            "status": "ok",
            "message": "Config reloaded successfully",
            "collections": list(collections.keys()),
            "count": len(collections)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ============== CẤU HÌNH ==============


import asyncio

@app.on_event("startup")
async def startup_event():
    """
    Khởi động luồng quét tự động (Internal Scheduler).
    Tự động quét 3 phút/lần.
    """
    asyncio.create_task(run_auto_scan_loop())

async def run_auto_scan_loop():
    # Lấy cấu hình từ .env (Mặc định 180s = 3 phút)
    interval = int(os.getenv("SYNC_INTERVAL_SECONDS", 180))
    print(f"⏰ [Internal Scheduler] Auto-Scan started (Every {interval}s)")
    
    from sync_dynamic import scan_new_databases
    while True:
        try:
            print(f"\n⏰ [Auto-Scan] Triggering scheduled scan (Next run in {interval}s)...")
            await scan_new_databases()
        except Exception as e:
            print(f"❌ [Auto-Scan] Error: {e}")
        
        await asyncio.sleep(interval)

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag", ".env")

try:
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH, override=True)
except Exception:
    pass

print("=== DEBUG ENV CHECK ===")
print("ENV_PATH =", ENV_PATH, "| exists:", os.path.exists(ENV_PATH))
print("NOTION_API_KEY =", os.getenv("NOTION_API_KEY"))
print("NOTION_BASE_URL =", os.getenv("NOTION_BASE_URL"))
print("DATABASE_ID_FAQ =", os.getenv("DATABASE_ID_FAQ"))
print("========================")

_notion_cached = None
_notion_warned_once = False  # chỉ cảnh báo 1 lần khi lỗi HTTP push

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_DB_PATH = os.path.join(BASE_DIR, "chat.db")
print(f"[ChatDB] Using: {CHAT_DB_PATH}")
DB_PATH = CHAT_DB_PATH

FAQ_DB_PATH = os.path.join(BASE_DIR, "faq.db")
CONF_THRESHOLD = 0.60
LOG_ALL_QUESTIONS = True

INTERRUPT_INTENTS = set()

# ============== DB helpers ==============
def ensure_main_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True) if os.path.dirname(DB_PATH) else None
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_message TEXT,
            bot_reply   TEXT,
            intent_tag  TEXT,
            confidence  REAL,
            time        TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    # Migration: check if session_id exists
    try:
        cur.execute("SELECT session_id FROM conversations LIMIT 1")
    except sqlite3.OperationalError:
        print("⚠️ [DB Fix] Adding missing column 'session_id'...")
        try:
            cur.execute("ALTER TABLE conversations ADD COLUMN session_id TEXT DEFAULT 'default'")
            print("✅ Added 'session_id' column successfully.")
        except Exception as e:
            print(f"❌ Failed to add column session_id: {e}")
            
    # Migration: check if created_at exists
    try:
        cur.execute("SELECT created_at FROM conversations LIMIT 1")
    except sqlite3.OperationalError:
        print("⚠️ [DB Fix] Adding missing column 'created_at'...")
        try:
            # SQLite limitation: Cannot ADD COLUMN with non-constant DEFAULT CURRENT_TIMESTAMP
            # So we add as TEXT (nullable), app handles the value insertion.
            cur.execute("ALTER TABLE conversations ADD COLUMN created_at TEXT")
            print("✅ Added 'created_at' column successfully.")
        except Exception as e:
            print(f"❌ Failed to add column created_at: {e}")

    conn.commit()
    return conn
def ensure_questions_log_db() -> None:
    dir_name = os.path.dirname(FAQ_DB_PATH)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    conn2 = sqlite3.connect(FAQ_DB_PATH)
    cur2 = conn2.cursor()
    cur2.execute(
        """
        CREATE TABLE IF NOT EXISTS questions_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question   TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            synced     INTEGER DEFAULT 0
        )
        """
    )
    conn2.commit()
    conn2.close()
def log_question_for_notion(question: str) -> None:
    if not question or not question.strip():
        return
    ensure_questions_log_db()
    conn2 = sqlite3.connect(FAQ_DB_PATH)
    cur2 = conn2.cursor()
    cur2.execute(
        "INSERT INTO questions_log (question, synced) VALUES (?, 0)",
        (question.strip(),),
    )
    conn2.commit()
    conn2.close()

def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def process_message(sentence: str, session_id: str = "default", image_path: str = None) -> str:
    sentence = (sentence or "").strip()

    # 0) Xử lý ảnh (OCR) nếu có
    if image_path:
        from ocr_helper import ocr_from_image
        ocr_text = ocr_from_image(image_path)
        if ocr_text:
            print(f"[PROCESS] OCR Result: {ocr_text}")
            if sentence:
                sentence = f"{sentence}\n\n[Nội dung từ ảnh]: {ocr_text}"
            else:
                sentence = f"[Nội dung từ ảnh]: {ocr_text}"

    if not sentence:
        reply = "Xin chào 👋 Bạn muốn hỏi thông tin gì trong thư viện?"
        tag_to_log = None
        confidence = 0.0
    else:
        # 1) Lấy lịch sử hội thoại (2-3 câu gần nhất, trong 10 phút)
        history = get_recent_history(session_id, limit=3, expire_minutes=10)
        
        if history:
            print(f"[MEMORY] Loaded {len(history)} previous messages for session: {session_id}")
        
        # 2) GỌI NÃO CHÍNH Ở FILE chat.py (với history)
        try:
            reply = chat.process_message(sentence, history=history, image_path=image_path)
        except Exception as e:
            print("[chat_fixed] Lỗi gọi chat.process_message:", e)
            reply = "Hiện tại hệ thống đang gặp lỗi khi xử lý câu hỏi của bạn."
        tag_to_log = None   # nếu sau này muốn lưu intent/category riêng thì sửa ở đây
        confidence = 1.0

    # 3) Ghi SQLite với session_id
    conn = ensure_main_db()
    cur  = conn.cursor()
    now_str = _now()
    cur.execute(
        "INSERT INTO conversations(session_id, user_message, bot_reply, intent_tag, confidence, time, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (session_id, sentence, reply, tag_to_log, confidence, now_str, now_str),
    )
    conn.commit()
    conn.close()

    # 3.1) Ghi thêm vào faq.db (inbox)
    try:
        log_question_for_notion(f"User: {sentence}\nBot: {reply}")
    except Exception as e:
        print(f"[Notion inbox] Lỗi ghi faq.db: {e}")

    # 4) Đẩy Notion (không chặn luồng chat)
    should_push = (
        LOG_ALL_QUESTIONS
        or reply.strip().startswith("Xin lỗi, mình chưa hiểu")
        or confidence < CONF_THRESHOLD
        or tag_to_log is None
    )
    if should_push:
        try:
            threading.Thread(
                target=push_to_notion,
                args=(sentence, reply),
                daemon=True
            ).start()
        except Exception as e:
            print("Notion push error:", e)

    return reply


# ============== CHAT ENDPOINT ==============
@app.post("/chat")
async def chat_endpoint(request: Request):
    """
    Main chat endpoint - accepts message and session_id from web interface
    """
    try:
        form = await request.form()
        message = form.get("message", "").strip()
        session_id = form.get("session_id", "default")
        
        if not message:
            return {"answer": "Xin chào 👋 Bạn muốn hỏi thông tin gì trong thư viện?"}
        
        # Process message with session context
        reply = process_message(message, session_id=session_id)
        
        return {"answer": reply}
    
    except Exception as e:
        print(f"[/chat] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"answer": "Xin lỗi, hệ thống đang gặp lỗi. Vui lòng thử lại."}



def _dns_ok(host: str, timeout_s: float = 3.0) -> bool:
    try:
        socket.setdefaulttimeout(timeout_s)
        socket.getaddrinfo(host, 443)
        return True
    except Exception:
        return False
def pull_approved_from_notion_to_sqlite():
    token, dbid, mode, base = _resolve_notion_env()
    url = f"{base.rstrip('/')}/databases/{dbid}/query"
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": os.getenv("NOTION_VERSION", "2022-06-28"),
        "Content-Type": "application/json",
    }
    body = {
        "filter": {
            "and": [
                {"property": "Approved", "checkbox": {"equals": True}},
            ]
        }
    }
    r = requests.post(url, headers=headers, json=body, timeout=12)
    r.raise_for_status()
    data = r.json()

    conn = ensure_main_db()
    cur = conn.cursor()

    for row in data.get("results", []):
        props = row.get("properties", {})
        q = props.get("Question", {}).get("rich_text", [{}])[0].get("plain_text", "")
        a = props.get("Answer", {}).get("rich_text", [{}])[0].get("plain_text", "")

        # lưu vào SQLite (ví dụ conversations hay bảng riêng)
        cur.execute(
            "INSERT INTO conversations(user_message, bot_reply, intent_tag, confidence, time) VALUES (?,?,?,?,?)",
            (q, a, None, 1.0, _now()),
        )
    conn.commit()
    conn.close()

# ============== Notion helpers (ntn_ token, auto-mapping) ==============
from functools import lru_cache

def _resolve_notion_env():
    try:
        if os.path.exists(ENV_PATH):
            load_dotenv(ENV_PATH, override=True)
    except Exception:
        pass
    token = os.getenv("NOTION_TOKEN") or os.getenv("NOTION_API_KEY") or ""
    dbid  = (
        os.getenv("NOTION_DATABASE_ID")
        or os.getenv("DATABASE_ID_FAQ")
        or os.getenv("DATABASE_ID_BOOKS")
        or os.getenv("DATABASE_ID_MAJORS")
        or ""
    )
    base  = (os.getenv("NOTION_BASE_URL") or "https://api.notion.com/v1").rstrip("/")
    mode  = "sdk" if token.startswith("secret_") else "http"  # ntn_ => http

    # Fallback an toàn nếu đang trỏ tới ntn-api nhưng DNS/route hỏng
    if token.startswith("ntn_") and "ntn-api.notion.so" in base:
        if not _dns_ok("ntn-api.notion.so"):
            base = "https://api.notion.com/v1"

    return token, dbid, mode, base

def _rt(txt: str):
    return [{"type": "text", "text": {"content": txt or ""}}]

def _http_session_with_retry(total=2, backoff=0.6):
    s = requests.Session()
    retry = Retry(
        total=total,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST", "HEAD"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

def _ntn_session():
    # nhẹ hơn, ưu tiên giảm chờ
    return _http_session_with_retry(total=1, backoff=0.4)

def ntn_ok(base: str) -> bool:
    """Preflight: confirm CF/Notion phản hồi để tránh timeout kéo dài."""
    base = (base or "").rstrip("/")
    try:
        r = requests.get("https://api.notion.com/v1/status", timeout=6)
        if r.status_code not in (200, 400, 401, 405):
            print("[Preflight] api.notion.com status:", r.status_code)
    except requests.exceptions.RequestException:
        return False

    if "ntn-api.notion.so" in base:
        try:
            rr = requests.head(f"{base}/pages", timeout=6)
            return rr.status_code in (200,201,400,401,403,405,429,500,502,503,504,530)
        except requests.exceptions.RequestException:
            return False
    return True

def _http_create_page(token: str, base_url: str, payload: dict, timeout_s: float = 15.0):
    """POST /pages, trả (ok, status, body_text)."""
    url = f"{base_url.rstrip('/')}/pages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
        "Notion-Version": os.getenv("NOTION_VERSION", "2022-06-28"),
        "Host": "ntn-api.notion.so" if "ntn-api.notion.so" in base_url else "api.notion.com",
    }
    try:
        sess = _ntn_session()
        r = sess.post(url, headers=headers, json=payload, timeout=timeout_s, allow_redirects=True)
        ok = r.status_code in (200, 201)
        return ok, r.status_code, r.text
    except requests.exceptions.Timeout:
        return False, 408, "timeout"
    except Exception as e:
        return False, -1, f"{type(e).__name__}: {e}"

@lru_cache(maxsize=8)
def _fetch_db_schema(token: str, base: str, dbid: str) -> dict:
    """Lấy schema DB để auto-map properties (cache theo (token,base,dbid))."""
    url = f"{base.rstrip('/')}/databases/{dbid}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": os.getenv("NOTION_VERSION", "2022-06-28"),
        "Accept": "application/json",
    }
    sess = _http_session_with_retry(total=2, backoff=0.5)
    r = sess.get(url, headers=headers, timeout=10)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"GET /databases/{dbid} FAIL {r.status_code}: {r.text[:500]}")
    return r.json()

def _pick_prop_by_type(props: dict, want_type: str, prefer_names: list[str]) -> Optional[str]:
    """Chọn tên cột theo type: ưu tiên theo danh sách tên gợi ý, fallback cột bất kỳ cùng type."""
    # ưu tiên theo tên
    lower_props = {k.lower(): k for k in props.keys()}
    for name in prefer_names:
        key = lower_props.get(name.lower())
        if key and props.get(key, {}).get("type") == want_type:
            return key
    # fallback: lấy cột đầu tiên có type phù hợp
    for k, v in props.items():
        if v.get("type") == want_type:
            return k
    return None
def _ensure_select_option(token: str, base: str, dbid: str, prop_name: str, option_name: str) -> str:
    """
    Đảm bảo option select tồn tại; nếu chưa có sẽ thêm (best effort).
    Trả lại option_name (có thể đã tồn tại hoặc vừa tạo).
    """
    # Đọc schema
    schema = _fetch_db_schema(token, base, dbid)
    props = schema.get("properties", {})
    prop = props.get(prop_name, {})
    if prop.get("type") != "select":
        return option_name  # không phải select thì bỏ qua

    options = prop.get("select", {}).get("options", []) or []
    names = {opt.get("name"): opt.get("id") for opt in options if isinstance(opt, dict)}
    if option_name in names:
        return option_name

    # Thử thêm option qua update database
    url = f"{base.rstrip('/')}/databases/{dbid}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Notion-Version": os.getenv("NOTION_VERSION", "2022-06-28"),
    }
    new_opt = {"name": option_name}
    body = {
        "properties": {
            prop_name: {
                "select": {
                    "options": options + [new_opt]
                }
            }
        }
    }
    try:
        r = requests.patch(url, headers=headers, json=body, timeout=12)
        if r.status_code in (200, 201):
            return option_name
        else:
            # Không tạo được option → vẫn dùng tên option (Notion sẽ reject nếu chưa có)
            print(f"[Notion] WARN: add select option FAIL {r.status_code}: {r.text[:400]}")
            return option_name
    except Exception as e:
        print(f"[Notion] WARN: add select option error: {e}")
        return option_name
def _build_dynamic_payload_force(dbid: str, q: str, a: str) -> dict:
    title_txt = (q or "Câu hỏi").strip()[:200]
    today_iso = datetime.now().date().isoformat()


    props = {
        "Question": {"rich_text": [{"type": "text", "text": {"content": q or ""}}]},
        "Answer":   {"rich_text": [{"type": "text", "text": {"content": a or ""}}]},
        # Cho item xuất hiện ngay ở view chính:
        "Approved": {"checkbox": True},  # <-- bật nếu view đang lọc Approved = checked
        "Language": {"select": {"name": "Tiếng Việt"}},  # <-- khớp filter Language
        "Last Update": {"date": {"start": today_iso}},
    }

    # Nếu bảng của bạn BẮT BUỘC có Category để vào view, set thêm 1 value hợp lệ:
    # props["Category"] = {"select": {"name": "Quy định"}}

    return {
        "parent": {"database_id": dbid},
        "properties": props,
    }
def push_to_notion(q: str, a: str):
    """
    Đẩy ngay từng dòng lên Notion (ntn_). Tự dò schema và map properties.
    In lỗi chi tiết khi fail để bạn sửa đúng chỗ.
    """
    global _notion_warned_once
    q = (q or "").strip(); a = (a or "").strip()
    if not q:
        return

    token, dbid, mode, base = _resolve_notion_env()
    if not token or not dbid:
        print("[Notion] Bỏ qua: thiếu token/dbid.")
        return

    # Chỉ hỗ trợ http (ntn_) ở đây; nếu bạn dùng secret_, có thể nhánh SDK.
    if mode != "http":
        print("[Notion] Bạn đang dùng secret_; nhánh HTTP này dành cho ntn_.")
        return
    # Preflight – tránh đợi timeout vô ích
    # 👉 Preflight: có thể BỎ QUA nếu FORCE_PUSH_NOTION=1
    force_push = os.getenv("FORCE_PUSH_NOTION", "0") == "1"
    if not force_push and not ntn_ok(base):
        if not _notion_warned_once:
            print("[Notion] Gateway hiện không reachable → bỏ qua lần này.")
            _notion_warned_once = True
        return
    else:
        if force_push:
            print("[Notion] FORCE: bỏ qua preflight, thử push trực tiếp...")


    # Build payload theo schema thực tế
    try:
        payload = _build_dynamic_payload_force(dbid, q, a)
    except Exception as e:
        print(f"[Notion] Build payload error: {e}")
        return

    ok, status, body = _http_create_page(token, base, payload, timeout_s=15.0)
    if ok:
        print(f"[Notion] OK ({status})")
    else:
        # In body đầy đủ để thấy lỗi thật (property nào sai type/tên/option)
        print(f"[Notion] FAIL ({status})\n{body[:2000]}")


def _ntn_session():    
    s = requests.Session()
    retry = Retry(
        total=1,               # chỉ 1 lần retry nhẹ để không chờ lâu
        backoff_factor=0.4,
        status_forcelist=[429,500,502,503,504],
        allowed_methods=["POST", "HEAD"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

def get_recent_history(session_id: str = None, limit=3, expire_minutes=10):
    """
    Lấy lịch sử hội thoại gần nhất theo session.
    
    Args:
        session_id: ID của session (từ LocalStorage)
        limit: Số câu tối đa (mặc định 3)
        expire_minutes: Thời gian hết hạn (mặc định 10 phút)
    
    Returns:
        List of (user_message, bot_reply) tuples
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        # Tính thời gian cutoff (10 phút trước)
        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(minutes=expire_minutes)).isoformat()
        
        if session_id:
            # Lọc theo session và thời gian
            cur.execute("""
                SELECT user_message, bot_reply
                FROM conversations
                WHERE session_id = ? 
                  AND datetime(created_at) > datetime(?)
                ORDER BY id DESC
                LIMIT ?
            """, (session_id, cutoff, limit))
        else:
            # Fallback: Lấy tất cả (backward compatible)
            cur.execute("""
                SELECT user_message, bot_reply
                FROM conversations
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
        
        rows = cur.fetchall()
        conn.close()
        # Đảo lại cho thành cũ → mới
        rows.reverse()
        return rows
    except Exception as e:
        print(f"[get_recent_history] Error: {e}")
        return []

if __name__ == "__main__":
    print("🤖 Chatbot 4-BƯỚC (Phiên bản TỐI ƯU RAM) đã sẵn sàng!")
    conn = ensure_main_db()
    cur  = conn.cursor()
    #_test_push_notion_once()
    try:
        while True:
            # Check for non-interactive mode
            if not sys.stdin.isatty():
                 # Keep the main thread alive for the web server
                 time.sleep(3600)
                 continue
                 
            q = input("\nBạn: ")
            if q.lower() in ["quit", "bye", "exit", "thoát"]:
                print("Hẹn gặp lại bạn ở thư viện nhé! 📚")
                break
            print("Bot:", process_message(q))
    finally:
        conn.close()