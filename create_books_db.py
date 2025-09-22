import sqlite3

# Kết nối đến DB (sử dụng faq.db để tích hợp chung)
conn = sqlite3.connect('faq.db')
c = conn.cursor()

# Tạo bảng majors nếu chưa tồn tại
c.execute('''
    CREATE TABLE IF NOT EXISTS majors (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT
    )
''')

# Tạo bảng books nếu chưa tồn tại
c.execute('''
    CREATE TABLE IF NOT EXISTS books (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        author TEXT,
        year INTEGER,
        quantity INTEGER,
        major_id INTEGER,
        status TEXT,
        link TEXT,
        available TEXT,
        FOREIGN KEY (major_id) REFERENCES majors(id)
    )
''')

# Chèn dữ liệu majors
majors = [
    (1, 'Công nghệ thông tin', None),
    (2, 'Kinh tế', None),
    (3, 'Y học', None),
    (4, 'Giáo dục Mầm non', None),
    (5, 'Giáo dục Tiểu học', None),
    (6, 'Giáo dục Tiểu học - Tiếng Jrai', None),
    (7, 'Giáo dục Chính trị', None),
    (8, 'Sư phạm Toán học', None),
    (9, 'Sư phạm Vật lý', None),
    (10, 'Sư phạm Hóa học', None)
]
c.executemany("INSERT OR IGNORE INTO majors (id, name, description) VALUES (?, ?, ?)", majors)

# Chèn dữ liệu books (sử dụng executemany để chèn hàng loạt)
books = [
    ('Cấu trúc dữ liệu & Giải thuật', 'Nguyễn Văn A', 2020, 12, 1, 'Còn hàng', 'https://www.notion.so/26f57231d19d80739d01ec90bd35c24b', 'Yes'),
    ('Trí tuệ nhân tạo', 'Trần Văn B', 2021, 8, 1, 'Còn hàng', 'https://www.notion.so/26f57231d19d80739d01ec90bd35c24b', 'Yes'),
    ('Lập trình Python', 'Lê Văn C', 2019, 15, 1, 'Còn hàng', 'https://www.notion.so/26f57231d19d80739d01ec90bd35c24b', 'Yes'),
    ('An toàn thông tin', 'Nguyễn Văn D', 2022, 10, 1, 'Còn hàng', 'https://www.notion.so/26f57231d19d80739d01ec90bd35c24b', 'Yes'),
    ('Hệ điều hành', 'Phạm Văn E', 2018, 6, 1, 'Còn hàng', 'https://www.notion.so/26f57231d19d80739d01ec90bd35c24b', 'Yes'),
    ('Mạng máy tính', 'Hoàng Văn F', 2019, 14, 1, 'Còn hàng', 'https://www.notion.so/26f57231d19d80739d01ec90bd35c24b', 'Yes'),
    ('Lập trình Web', 'Nguyễn Văn G', 2021, 9, 1, 'Còn hàng', 'https://www.notion.so/26f57231d19d80739d01ec90bd35c24b', 'Yes'),
    ('Khoa học dữ liệu', 'Trần Văn H', 2020, 13, 1, 'Còn hàng', 'https://www.notion.so/26f57231d19d80739d01ec90bd35c24b', 'Yes'),
    ('Machine Learning', 'Phạm Văn I', 2022, 7, 1, 'Còn hàng', 'https://www.notion.so/26f57231d19d80739d01ec90bd35c24b', 'Yes'),
    ('Cơ sở dữ liệu', 'Nguyễn Văn K', 2019, 11, 1, 'Còn hàng', 'https://www.notion.so/26f57231d19d80739d01ec90bd35c24b', 'Yes'),
    ('Kinh tế học vi mô', 'Nguyễn Thị A', 2018, 10, 2, 'Còn hàng', 'https://www.notion.so/26f57231d19d802499bdd0b48ec77f6c', 'Yes'),
    ('Kinh tế học vĩ mô', 'Trần Thị B', 2019, 8, 2, 'Còn hàng', 'https://www.notion.so/26f57231d19d802499bdd0b48ec77f6c', 'Yes'),
    ('Marketing căn bản', 'Lê Thị C', 2020, 14, 2, 'Còn hàng', 'https://www.notion.so/26f57231d19d802499bdd0b48ec77f6c', 'Yes'),
    ('Quản trị kinh doanh', 'Nguyễn Văn D', 2021, 7, 2, 'Còn hàng', 'https://www.notion.so/26f57231d19d802499bdd0b48ec77f6c', 'Yes'),
    ('Tài chính doanh nghiệp', 'Phạm Văn E', 2019, 15, 2, 'Còn hàng', 'https://www.notion.so/26f57231d19d802499bdd0b48ec77f6c', 'Yes'),
    ('Kế toán tài chính', 'Hoàng Văn F', 2018, 6, 2, 'Còn hàng', 'https://www.notion.so/26f57231d19d802499bdd0b48ec77f6c', 'Yes'),
    ('Phân tích đầu tư', 'Nguyễn Văn G', 2022, 12, 2, 'Còn hàng', 'https://www.notion.so/26f57231d19d802499bdd0b48ec77f6c', 'Yes'),
    ('Quản trị nhân sự', 'Trần Văn H', 2021, 9, 2, 'Còn hàng', 'https://www.notion.so/26f57231d19d802499bdd0b48ec77f6c', 'Yes'),
    ('Thương mại điện tử', 'Phạm Văn I', 2020, 11, 2, 'Còn hàng', 'https://www.notion.so/26f57231d19d802499bdd0b48ec77f6c', 'Yes'),
    ('Kinh doanh quốc tế', 'Nguyễn Văn K', 2022, 13, 2, 'Còn hàng', 'https://www.notion.so/26f57231d19d802499bdd0b48ec77f6c', 'Yes'),
    ('Giải phẫu học', 'Nguyễn Thị A', 2018, 12, 3, 'Còn hàng', 'https://www.notion.so/26f57231d19d80ce9c7cdf794eb8ecb4', 'Yes'),
    ('Sinh lý học', 'Trần Thị B', 2019, 8, 3, 'Còn hàng', 'https://www.notion.so/26f57231d19d80ce9c7cdf794eb8ecb4', 'Yes'),
    ('Dược lý học', 'Lê Thị C', 2020, 14, 3, 'Còn hàng', 'https://www.notion.so/26f57231d19d80ce9c7cdf794eb8ecb4', 'Yes'),
    ('Chẩn đoán hình ảnh', 'Nguyễn Văn D', 2021, 7, 3, 'Còn hàng', 'https://www.notion.so/26f57231d19d80ce9c7cdf794eb8ecb4', 'Yes'),
    ('Nội khoa cơ bản', 'Phạm Văn E', 2019, 10, 3, 'Còn hàng', 'https://www.notion.so/26f57231d19d80ce9c7cdf794eb8ecb4', 'Yes'),
    ('Ngoại khoa cơ bản', 'Hoàng Văn F', 2018, 6, 3, 'Còn hàng', 'https://www.notion.so/26f57231d19d80ce9c7cdf794eb8ecb4', 'Yes'),
    ('Sản phụ khoa', 'Nguyễn Văn G', 2022, 15, 3, 'Còn hàng', 'https://www.notion.so/26f57231d19d80ce9c7cdf794eb8ecb4', 'Yes'),
    ('Nhi khoa', 'Trần Văn H', 2021, 9, 3, 'Còn hàng', 'https://www.notion.so/26f57231d19d80ce9c7cdf794eb8ecb4', 'Yes'),
    ('Y học cổ truyền', 'Phạm Văn I', 2020, 11, 3, 'Còn hàng', 'https://www.notion.so/26f57231d19d80ce9c7cdf794eb8ecb4', 'Yes'),
    ('Dinh dưỡng học', 'Nguyễn Văn K', 2022, 13, 3, 'Còn hàng', 'https://www.notion.so/26f57231d19d80ce9c7cdf794eb8ecb4', 'Yes'),
    ('Phương pháp tổ chức hoạt động vui chơi cho trẻ mầm non', 'Nguyễn Thị Lan', 2018, 12, 4, 'Còn hàng', 'https://www.notion.so/26f57231d19d80288b89c79f52019b8b', 'Yes'),
    ('Tâm lý học trẻ em lứa tuổi mầm non', 'Trần Văn Minh', 2019, 9, 4, 'Còn hàng', 'https://www.notion.so/26f57231d19d80288b89c79f52019b8b', 'Yes'),
    ('Giáo dục âm nhạc cho trẻ mầm non', 'Lê Thị Hoa', 2020, 8, 4, 'Còn hàng', 'https://www.notion.so/26f57231d19d80288b89c79f52019b8b', 'Yes'),
    ('Phát triển ngôn ngữ cho trẻ mầm non', 'Phạm Thị Hương', 2017, 10, 4, 'Còn hàng', 'https://www.notion.so/26f57231d19d80288b89c79f52019b8b', 'Yes'),
    ('Giáo dục thể chất cho trẻ mầm non', 'Nguyễn Văn Thành', 2018, 6, 4, 'Còn hàng', 'https://www.notion.so/26f57231d19d80288b89c79f52019b8b', 'Yes'),
    ('Chăm sóc sức khỏe trẻ mầm non', 'Nguyễn Thị Thu', 2021, 11, 4, 'Còn hàng', 'https://www.notion.so/26f57231d19d80288b89c79f52019b8b', 'Yes'),
    ('Tổ chức hoạt động tạo hình cho trẻ mầm non', 'Vũ Thị Mai', 2019, 5, 4, 'Còn hàng', 'https://www.notion.so/26f57231d19d80288b89c79f52019b8b', 'Yes'),
    ('Phát triển tình cảm – kỹ năng xã hội cho trẻ mầm non', 'Lê Văn Dũng', 2020, 13, 4, 'Còn hàng', 'https://www.notion.so/26f57231d19d80288b89c79f52019b8b', 'Yes'),
    ('Giáo dục lễ giáo cho trẻ mầm non', 'Hoàng Thị Thanh', 2022, 14, 4, 'Còn hàng', 'https://www.notion.so/26f57231d19d80288b89c79f52019b8b', 'Yes'),
    ('Tổ chức hoạt động khám phá khoa học cho trẻ mầm non', 'Nguyễn Thị Oanh', 2021, 20, 4, 'Còn hàng', 'https://www.notion.so/26f57231d19d80288b89c79f52019b8b', 'Yes'),
    ('Phương pháp dạy học Toán ở tiểu học', 'Nguyễn Văn An', 2018, 8, 5, 'Còn hàng', 'https://www.notion.so/26f57231d19d8008b35fd01c6693924a', 'Yes'),
    ('Phương pháp dạy học Tiếng Việt ở tiểu học', 'Trần Thị Bình', 2019, 9, 5, 'Còn hàng', 'https://www.notion.so/26f57231d19d8008b35fd01c6693924a', 'Yes'),
    ('Giáo dục đạo đức ở tiểu học', 'Lê Văn Cường', 2020, 7, 5, 'Còn hàng', 'https://www.notion.so/26f57231d19d8008b35fd01c6693924a', 'Yes'),
    ('Phương pháp dạy học Tự nhiên và Xã hội', 'Nguyễn Thị Hạnh', 2017, 11, 5, 'Còn hàng', 'https://www.notion.so/26f57231d19d8008b35fd01c6693924a', 'Yes'),
    ('Tâm lý học lứa tuổi tiểu học', 'Phạm Thị Mai', 2018, 10, 5, 'Còn hàng', 'https://www.notion.so/26f57231d19d8008b35fd01c6693924a', 'Yes'),
    ('Đánh giá kết quả học tập ở tiểu học', 'Nguyễn Văn Hoàng', 2021, 6, 5, 'Còn hàng', 'https://www.notion.so/26f57231d19d8008b35fd01c6693924a', 'Yes'),
    ('Phát triển kỹ năng sống cho học sinh tiểu học', 'Vũ Thị Thu', 2019, 15, 5, 'Còn hàng', 'https://www.notion.so/26f57231d19d8008b35fd01c6693924a', 'Yes'),
    ('Ứng dụng CNTT trong dạy học tiểu học', 'Lê Thị Thanh', 2022, 5, 5, 'Còn hàng', 'https://www.notion.so/26f57231d19d8008b35fd01c6693924a', 'Yes'),
    ('Giáo dục thể chất ở tiểu học', 'Nguyễn Văn Tuấn', 2020, 13, 5, 'Còn hàng', 'https://www.notion.so/26f57231d19d8008b35fd01c6693924a', 'Yes'),
    ('Phương pháp dạy học Âm nhạc ở tiểu học', 'Hoàng Thị Nga', 2021, 17, 5, 'Còn hàng', 'https://www.notion.so/26f57231d19d8008b35fd01c6693924a', 'Yes')
]
c.executemany("INSERT INTO books (name, author, year, quantity, major_id, status, link, available) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", books)

# Commit và đóng kết nối
conn.commit()
conn.close()

print("DB đã được tạo và dữ liệu đã được chèn thành công!")

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

