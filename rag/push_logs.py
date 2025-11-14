import os
import sys
import sqlite3
import argparse
from datetime import datetime
from dotenv import load_dotenv
from notion_client import Client

# =========================
# CẤU HÌNH ĐƯỜNG DẪN
# =========================

ENV_PATH = r"D:\HTML\a\rag\.env"
     # chứa NOTION_TOKEN, NOTION_DATABASE_ID
CHAT_DB   = r"D:\HTML\a\chat.db"
# Lưu ý: KHÔNG dùng faqs.db nữa trong script này

# =========================
# TIỆN ÍCH
# =========================
def rt(txt: str):
    """Rich text Notion helper."""
    return [{"type": "text", "text": {"content": txt or ""}}]

def ensure_sync_table():
    """Tạo bảng theo dõi sync trong chat.db nếu chưa có."""
    con = sqlite3.connect(CHAT_DB)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations_sync (
        conv_id INTEGER PRIMARY KEY,
        notion_page_id TEXT,
        synced INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    con.commit()
    con.close()

def load_env():
    if not os.path.exists(ENV_PATH):
        print(f"❌ Không tìm thấy file .env tại: {ENV_PATH}")
        sys.exit(1)
    load_dotenv(ENV_PATH)
    token = os.getenv("NOTION_TOKEN")
    dbid  = os.getenv("NOTION_DATABASE_ID")
    if not token or not dbid:
        print("❌ Thiếu NOTION_TOKEN hoặc NOTION_DATABASE_ID trong .env")
        sys.exit(1)
    return token, dbid

def fetch_rows(limit=None, force_all=False):
    """
    Lấy các dòng từ conversations.
    - Nếu force_all=True: lấy TẤT CẢ.
    - Ngược lại: chỉ lấy những dòng CHƯA sync (không có bản hldf ghi trong conversations_sync, hoặc synced=0).
    """
    con = sqlite3.connect(CHAT_DB)
    cur = con.cursor()
    if force_all:
        sql = """
            SELECT c.id, c.user_message, c.bot_reply
            FROM conversations c
            ORDER BY c.id ASC
        """
        params = ()
    else:
        sql = """
            SELECT c.id, c.user_message, c.bot_reply
            FROM conversations c
            LEFT JOIN conversations_sync s ON s.conv_id = c.id
            WHERE s.conv_id IS NULL OR s.synced = 0
            ORDER BY c.id ASC
        """
        params = ()
    if limit:
        sql += " LIMIT ?"
        params = params + (int(limit),)
    rows = cur.execute(sql, params).fetchall()
    con.close()
    return rows

def mark_synced(conv_id: int, page_id: str):
    """Đánh dấu 1 dòng conversations đã sync trong conversations_sync."""
    con = sqlite3.connect(CHAT_DB)
    cur = con.cursor()
    # INSERT or REPLACE để idempotent
    cur.execute("""
        INSERT INTO conversations_sync (conv_id, notion_page_id, synced, created_at)
        VALUES (?, ?, 1, CURRENT_TIMESTAMP)
        ON CONFLICT(conv_id) DO UPDATE SET
            notion_page_id=excluded.notion_page_id,
            synced=1,
            created_at=CURRENT_TIMESTAMP
    """, (conv_id, page_id))
    con.commit()
    con.close()

def reset_all_sync_flags():
    """Đặt lại tất cả conversation về trạng thái CHƯA sync trong bảng conversations_sync."""
    con = sqlite3.connect(CHAT_DB)
    cur = con.cursor()
    cur.execute("DELETE FROM conversations_sync")  # đơn giản nhất: xoá bảng theo dõi
    con.commit()
    con.close()

# =========================
# MAIN
# =========================



def main():
    parser = argparse.ArgumentParser(description="Push conversations (Question+Answer) lên Notion.")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số dòng cần đẩy")
    parser.add_argument("--force-all", action="store_true", help="Bỏ qua trạng thái sync, đẩy TẤT CẢ hội thoại")
    args = parser.parse_args()

    token, dbid = load_env()
    client = Client(auth=token)
    ensure_sync_table()

    if args.force_all:
        # Không bắt buộc, nhưng nếu muốn 'thật sự' sạch trạng thái:
        # reset_all_sync_flags()
        # Hoặc chỉ đơn giản bỏ qua bảng sync khi fetch (đã làm ở fetch_rows)
        pass

    rows = fetch_rows(limit=args.limit, force_all=args.force_all)
    if not rows:
        print("✅ Không có dòng hội thoại nào cần đẩy.")
        return

    pushed = 0
    for conv_id, user_msg, bot_reply in rows:
        q = (user_msg or "").strip()
        a = (bot_reply or "").strip()

        # Bỏ qua dòng trống (không có question)
        if len(q) == 0:
            mark_synced(conv_id, page_id="")  # đánh dấu để không lặp lại
            continue

        title = q[:200]  # Title nên ngắn
        try:
            page = client.pages.create(
                parent={"database_id": dbid},
                properties={
                    #"Tên":      {"title": rt(title)},
                    "Question": {"rich_text": rt(q)},
                    "Answer":   {"rich_text": rt(a)},
                    "Approved": {"checkbox": False},
                    "Language": {"select": {"name": "Tiếng Việt"}},
                }
            )
            page_id = page.get("id", "")
            mark_synced(conv_id, page_id)
            pushed += 1
        except Exception as e:
            # KHÔNG đánh dấu synced để lần sau thử lại
            print(f"⚠️ Lỗi khi tạo page cho conv_id={conv_id}: {e}")

    print(f"✅ Đã đẩy {pushed} dòng hội thoại (Question+Answer) lên Notion.")

if __name__ == "__main__":
    main()
