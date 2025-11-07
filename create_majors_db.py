#create_majors
import requests
import sqlite3
import logging
import os
from dotenv import load_dotenv   # THÊM DÒNG NÀY
load_dotenv()                    # THÊM DÒNG NÀY
# --- Cấu hình ---
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
DATABASE_ID = os.getenv("DATABASE_ID_MAJORS")
DB_PATH = os.path.normpath(r"C:\Users\ADMIN\OneDrive\Desktop\aaa\faq.db")
if not NOTION_API_KEY or not DATABASE_ID:
    raise ValueError("❌ Thiếu NOTION_API_KEY hoặc DATABASE_ID_MAJORS trong .env")
# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Headers Notion API ---
headers = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28"
}

# --- Khởi tạo DB ---
def khoi_tao_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS majors (
            id TEXT PRIMARY KEY,
            major_id INTEGER,
            name TEXT,
            description TEXT
        )
    """)
    conn.commit()
    return conn

# --- Lấy dữ liệu từ Notion ---
def lay_du_lieu_notion():
    url = f"https://api.notion.com/v1/databases/{DATABASE_ID}/query"
    payload = {}   # bỏ filter để chắc chắn lấy được
    res = requests.post(url, headers=headers, json=payload)
    print("DEBUG response:", res.text)  # in thẳng response để xem property
    res.raise_for_status()
    return res.json().get("results", [])

# --- Trích xuất thuộc tính ---
def trich_xuat_thuoc_tinh(row):
    props = row.get("properties", {})
    notion_id = row.get("id", "")

    # Cột Tên ngành (title)
    name = ""
    name_prop = props.get("Tên ngành", {})
    if name_prop.get("title"):
        name = name_prop["title"][0].get("plain_text", "")

    # Cột ID ngành
    major_id = ""
    id_prop = props.get("ID ngành", {})
    if id_prop.get("rich_text"):
        major_id = id_prop["rich_text"][0].get("plain_text", "")
    elif id_prop.get("number") is not None:
        major_id = str(id_prop["number"])

    # Cột Mô tả
    description = ""
    desc_prop = props.get("Mô tả", {})
    if desc_prop.get("rich_text"):
        description = desc_prop["rich_text"][0].get("plain_text", "")

    return notion_id, major_id, name, description

# --- Đồng bộ vào SQLite ---
def dong_bo_sqlite(du_lieu):
    conn = khoi_tao_db()
    cur = conn.cursor()
    for row in du_lieu:
        notion_id, major_id, name, description = trich_xuat_thuoc_tinh(row)
        if not notion_id or not name:
            logger.warning(f"Bỏ qua dòng thiếu dữ liệu: {row}")
            continue
        cur.execute("""
            INSERT OR REPLACE INTO majors (id, major_id, name, description)
            VALUES (?, ?, ?, ?)
        """, (notion_id, major_id, name, description))
        logger.info(f"Đã xử lý ngành: {name}")
    conn.commit()
    conn.close()

# --- Main ---
if __name__ == "__main__":
    data = lay_du_lieu_notion()
    dong_bo_sqlite(data)
    print("Hoàn tất đồng bộ majors!")
