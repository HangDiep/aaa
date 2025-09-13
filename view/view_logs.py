import sqlite3

DB_PATH = "chat.db"

def view_logs(limit=20):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, user_message, bot_reply, intent_tag, confidence, time "
        "FROM conversations ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("⚠️ Chưa có log nào trong chat.db")
        return

    print(f"📜 Hiển thị {len(rows)} log mới nhất:\n")
    for r in rows[::-1]:  # đảo ngược để in từ cũ -> mới
        id, user_msg, bot_msg, tag, conf, time = r
        tag = tag or "(None)"
        conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "N/A"
        print(f"[{id}] {time}")
        print(f"   Bạn: {user_msg}")
        print(f"   Bot ({tag}, {conf_str}): {bot_msg}\n")

if __name__ == "__main__":
    view_logs(limit=20)   # đổi số 20 nếu muốn nhiều hơn
