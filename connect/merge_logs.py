import sqlite3, json, os
from collections import defaultdict

DB_PATH = "chat.db"
INTENTS_PATH = "intents.json"        # file gốc của bạn
BACKUP_PATH = "intents.backup.json"  # an toàn: lưu backup

# 1) Đọc intents gốc
with open(INTENTS_PATH, "r", encoding="utf-8-sig") as f:
    intents_data = json.load(f)
existing = intents_data.get("intents", [])

# 2) Đọc log từ SQLite
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("SELECT user_message, bot_reply, COALESCE(intent_tag, 'from_logs') AS tag FROM conversations WHERE user_message IS NOT NULL AND bot_reply IS NOT NULL")
rows = cur.fetchall()
conn.close()

# 3) Gom theo tag
by_tag = defaultdict(lambda: {"patterns": set(), "responses": set()})
for user_msg, bot_msg, tag in rows:
    user_msg = user_msg.strip()
    bot_msg = bot_msg.strip()
    if user_msg:
        by_tag[tag]["patterns"].add(user_msg)
    if bot_msg:
        by_tag[tag]["responses"].add(bot_msg)

# 4) Chuyển set -> list + lọc rỗng
new_intents = []
for tag, pr in by_tag.items():
    patterns = sorted(p for p in pr["patterns"] if p)
    responses = sorted(r for r in pr["responses"] if r)
    if patterns and responses:
        new_intents.append({
            "tag": tag,
            "patterns": patterns,
            "responses": responses
        })

if not new_intents:
    print("Không có dữ liệu mới trong DB để merge.")
    raise SystemExit(0)

# 5) Hợp nhất: thêm/ghép vào intent trùng tag nếu đã có
def index_by_tag(intents_list):
    return {it["tag"]: it for it in intents_list if "tag" in it}

idx = index_by_tag(existing)

for ni in new_intents:
    tag = ni["tag"]
    if tag in idx:
        # ghép nối, loại trùng
        before_p = set(idx[tag].get("patterns", []))
        before_r = set(idx[tag].get("responses", []))
        idx[tag]["patterns"] = sorted(before_p.union(ni["patterns"]))
        idx[tag]["responses"] = sorted(before_r.union(ni["responses"]))
    else:
        existing.append(ni)

# 6) Ghi file (backup trước)
if os.path.exists(INTENTS_PATH):
    with open(BACKUP_PATH, "w", encoding="utf-8") as bf:
        json.dump(intents_data, bf, ensure_ascii=False, indent=2)
        print(f"Đã backup: {BACKUP_PATH}")

with open(INTENTS_PATH, "w", encoding="utf-8") as f:
    json.dump({"intents": existing}, f, ensure_ascii=False, indent=2)

print(f"✅ Merge xong vào {INTENTS_PATH}. Bạn có thể train lại bằng train.py.")
