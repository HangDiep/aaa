#create_books_db.py
import os
import sqlite3
import requests
import logging
from dotenv import load_dotenv
# =========================
# Config
# =========================
load_dotenv()
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
DATABASE_ID = os.getenv("DATABASE_ID_BOOKS") 
DB_PATH = "faq.db"

if not NOTION_API_KEY or not DATABASE_ID:
    raise ValueError("❌ Thiếu NOTION_API_KEY hoặc DATABASE_ID_BOOKS trong .env")

NOTION_URL = f"https://api.notion.com/v1/databases/{DATABASE_ID}/query"
HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28"
}

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# =========================
# DB Init
# =========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ten_sach TEXT,
            tac_gia TEXT,
            nam_xuat_ban INTEGER,
            so_luong_ton_kho INTEGER,
            nganh TEXT,
            id_nganh INTEGER,
            trang_thai TEXT,
            approved BOOLEAN
        )
    """)
    conn.commit()
    logging.info("Đã tạo hoặc xác nhận bảng books.")
    return conn

# =========================
# Helper: Lấy tên ngành từ Notion API
# =========================
def get_major_name(major_id: str):
    if not major_id:
        return None
    url = f"https://api.notion.com/v1/pages/{major_id}"
    res = requests.get(url, headers=HEADERS)
    if res.status_code != 200:
        logging.warning(f"Lỗi lấy tên ngành {major_id}: {res.text}")
        return None
    data = res.json()
    props = data.get("properties", {})
    name_prop = props.get("Tên ngành", {})
    if name_prop.get("title"):
        return name_prop["title"][0].get("plain_text", "")
    return None

# =========================
# Get data from Notion
# =========================
def fetch_books():
    books = []
    payload = {
        "page_size": 100,
        "filter": {
            "and": [
                {"property": "Tên sách", "title": {"is_not_empty": True}},
                {"property": "Ngành", "relation": {"is_not_empty": True}},
                {"property": "ID ngành", "rollup": {"any": {"number": {"is_not_empty": True}}}},
                {"property": "Trạng thái", "select": {"is_not_empty": True}}
            ]
        }
    }
    cursor = None

    while True:
        if cursor:
            payload["start_cursor"] = cursor

        response = requests.post(NOTION_URL, headers=HEADERS, json=payload)
        if response.status_code != 200:
            logging.error(f"Lỗi gọi API Notion: {response.status_code} - {response.text}")
            raise Exception(f"Notion API error: {response.status_code}")

        data = response.json()
        results = data.get("results", [])
        books.extend(results)
        cursor = data.get("next_cursor")

        logging.info(f"Đã lấy {len(results)} bản ghi, tổng cộng: {len(books)}")

        if not cursor or not data.get("has_more"):
            break

    if not books:
        logging.warning("Không có dữ liệu nào từ Notion.")
    return books

# =========================
# Save to DB
# =========================
def save_books(conn, books_data):
    cur = conn.cursor()
    cur.execute("DELETE FROM books")  # clear old data

    if books_data:
        logging.info(f"Các property trong Notion (mẫu): {list(books_data[0]['properties'].keys())}")

    for row in books_data:
        props = row["properties"]

        # Extract relation UUID
        nganh_relation = props.get("Ngành", {}).get("relation", [])
        nganh_uuid = nganh_relation[0].get("id") if nganh_relation else None

        # Lấy tên ngành từ API
        nganh_name = get_major_name(nganh_uuid) if nganh_uuid else None

        id_nganh_rollup = props.get("ID ngành", {}).get("rollup", {})
        id_nganh = id_nganh_rollup.get("number") if id_nganh_rollup.get("type") == "number" \
                   else (id_nganh_rollup.get("array", [{}])[0].get("number") if id_nganh_rollup.get("array") else None)

        trang_thai = props.get("Trạng thái", {}).get("select", {}).get("name")

        book = {
            "ten_sach": props.get("Tên sách", {}).get("title", [{}])[0].get("plain_text") 
                        if props.get("Tên sách", {}).get("title") else None,
            "tac_gia": props.get(" Tác giả", {}).get("rich_text", [{}])[0].get("plain_text")  
                        if props.get(" Tác giả", {}).get("rich_text") else None,
            "nam_xuat_ban": props.get("Năm xuất bản", {}).get("number"),
            "so_luong_ton_kho": props.get("Số lượng tồn kho", {}).get("number"),
            "nganh": nganh_name,  # ✅ Lưu thẳng tên ngành
            "id_nganh": id_nganh,
            "trang_thai": trang_thai,
            "approved": props.get("Approved", {}).get("checkbox")
        }

        # Log giá trị extract
        logging.info(f"Extracted: Ngành: {book['nganh']}, ID ngành: {book['id_nganh']}, Trạng thái: {book['trang_thai']}")

        if not book["ten_sach"] or not book["nganh"] or book["id_nganh"] is None or not book["trang_thai"]:
            logging.warning(f"Bỏ qua page thiếu dữ liệu: {row.get('id')} - Thiếu: {', '.join(k for k, v in book.items() if v is None)}")
            continue

        logging.info(f"Đang lưu sách: {book['ten_sach']}")

        cur.execute("""
            INSERT INTO books 
            (ten_sach, tac_gia, nam_xuat_ban, so_luong_ton_kho, nganh, id_nganh, trang_thai, approved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            book["ten_sach"],
            book["tac_gia"],
            book["nam_xuat_ban"],
            book["so_luong_ton_kho"],
            book["nganh"],
            book["id_nganh"],
            book["trang_thai"],
            book["approved"]
        ))

    conn.commit()
    logging.info("Đã đồng bộ xong dữ liệu Sách.")

# =========================
# Main
# =========================
if __name__ == "__main__":
    try:
        conn = init_db()
        books_data = fetch_books()
        save_books(conn, books_data)
        conn.close()
    except Exception as e:
        logging.error(f"Đồng bộ thất bại: {e}")
        print(f"Lỗi: {e}")
