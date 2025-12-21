# ==========================================
# HO TÊN: Đỗ Thị Hồng Điệp
# MSSV: 23103014
# ĐỒ ÁN: Chatbot Dynamic Router - TTN University
# NGÀY NỘP: 21/12/2025
# Copyright © 2025. All rights reserved.
# ==========================================

import sqlite3, json
from collections import defaultdict

conn = sqlite3.connect("chat.db")
cur = conn.cursor()
cur.execute("SELECT user_message, bot_reply, intent_tag FROM conversations WHERE user_message IS NOT NULL AND bot_reply IS NOT NULL")
rows = cur.fetchall()
conn.close()

by_tag = defaultdict(lambda: {"patterns": set(), "responses": set()})
for user_msg, bot_msg, tag in rows:
    tag = tag or "from_logs"   # nếu intent_tag trống
    by_tag[tag]["patterns"].add(user_msg.strip())
    by_tag[tag]["responses"].add(bot_msg.strip())

intents = []
for tag, pr in by_tag.items():
    intents.append({
        "tag": tag,
        "patterns": list(pr["patterns"]),
        "responses": list(pr["responses"])
    })

with open("intents_from_db.json", "w", encoding="utf-8") as f:
    json.dump({"intents": intents}, f, ensure_ascii=False, indent=2)

print("✅ Đã export xong sang intents_from_db.json")
