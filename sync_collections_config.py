import sqlite3

DB_PATH = "faq.db"

def get_existing_tables_from_notion():
    # TODO: Replace with real Notion tables when ready
    return ["faq_", "ngnh", "sch_"]


def sync_collections_config():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    notion_tables = set(get_existing_tables_from_notion())

    # Lấy danh sách bảng hiện có từ sqlite (cột name)
    cur.execute("SELECT name FROM collections_config")
    existing = set(row[0] for row in cur.fetchall())

    # Các bảng cần xóa
    to_delete = existing - notion_tables

    if not to_delete:
        print("✔ Không có bảng nào cần xóa. collections_config đã sạch.")
    else:
        print("🧹 Đang xóa các bảng không còn trong Notion:")
        for t in to_delete:
            cur.execute("DELETE FROM collections_config WHERE name = ?", (t,))
            print(f"   - Đã xóa: {t}")

    conn.commit()
    conn.close()
    print("\n🎉 Đồng bộ collections_config hoàn tất!")


if __name__ == "__main__":
    sync_collections_config()
