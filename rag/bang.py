import sqlite3

# Kết nối tới database (nếu chưa có thì sẽ tạo mới)
conn = sqlite3.connect("../faq.db")  # đổi đường dẫn nếu cần
cur = conn.cursor()

# Tạo lại bảng conversations
cur.execute("DROP TABLE IF EXISTS conversations;")
cur.execute("""
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_message TEXT,
    bot_reply TEXT,
    intent_tag TEXT,
    confidence REAL,
    time TEXT
);
""")

conn.commit()
conn.close()

print("✅ Bảng conversations đã được tạo")
