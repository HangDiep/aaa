
import requests
import sqlite3
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
# --- Cấu hình ---
load_dotenv()
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
DATABASE_ID = os.getenv("DATABASE_ID_FAQ") 
DB_PATH = "faq.db"

if not NOTION_API_KEY or not DATABASE_ID:
    raise ValueError("❌ Thiếu NOTION_API_KEY hoặc DATABASE_ID_FAQ trong .env")

# --- Thiết lập Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Tiêu đề API Notion ---
headers = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28"
}

# --- Khởi tạo cơ sở dữ liệu SQLite ---
def khoi_tao_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS faq (
                id TEXT PRIMARY KEY,
                question TEXT,
                answer TEXT,
                category TEXT,
                language TEXT,
                approved INTEGER,
                last_updated TEXT
            )
        """)
        conn.commit()
        logger.info("Khởi tạo cơ sở dữ liệu SQLite thành công.")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Lỗi khởi tạo cơ sở dữ liệu SQLite: {e}")
        raise

# --- Lấy dữ liệu từ Notion với bộ lọc ---
def lay_du_lieu_notion():
    url = f"https://api.notion.com/v1/databases/{DATABASE_ID}/query"
    payload = {
        "filter": {
            "and": [
                {"property": "Question", "rich_text": {"is_not_empty": True}},
                {"property": "Answer", "rich_text": {"is_not_empty": True}}
            ]
        }
    }
    try:
        res = requests.post(url, headers=headers, json=payload)
        if res.status_code == 401:
            logger.error("Xác thực thất bại. Vui lòng kiểm tra NOTION_TOKEN hoặc quyền truy cập tích hợp.")
            raise Exception("Lỗi 401: Xác thực thất bại. Kiểm tra NOTION_TOKEN và quyền truy cập.")
        if res.status_code == 404:
            logger.error("Không tìm thấy cơ sở dữ liệu. Vui lòng kiểm tra DATABASE_ID.")
            raise Exception("Lỗi 404: Cơ sở dữ liệu Notion không tồn tại.")
        res.raise_for_status()
        return res.json().get("results", [])
    except requests.RequestException as e:
        logger.error(f"Lỗi khi lấy dữ liệu từ Notion: {e}")
        raise

# --- Trích xuất thuộc tính an toàn ---
def trich_xuat_thuoc_tinh(row):
    props = row.get("properties", {})
    notion_id = row.get("id", "")
    
    # Trích xuất câu hỏi
    cau_hoi = ""
    question_prop = props.get("Question", {})
    if question_prop.get("title") and len(question_prop["title"]) > 0:
        cau_hoi = question_prop["title"][0].get("plain_text", "")
    elif question_prop.get("rich_text") and len(question_prop["rich_text"]) > 0:
        cau_hoi = question_prop["rich_text"][0].get("plain_text", "")
    
    # Trích xuất câu trả lời
    cau_tra_loi = ""
    answer_prop = props.get("Answer", {})
    if answer_prop.get("rich_text") and len(answer_prop["rich_text"]) > 0:
        cau_tra_loi = answer_prop["rich_text"][0].get("plain_text", "")
    
    # Trích xuất danh mục (xử lý nếu select là None)
    danh_muc = ""
    category_prop = props.get("Category", {})
    select_prop = category_prop.get("select")
    if select_prop is not None:
        danh_muc = select_prop.get("name", "")
    else:
        logger.warning(f"Dòng {notion_id} thiếu hoặc có select None trong Category.")
    
    # Trích xuất ngôn ngữ
    ngon_ngu = "Tiếng Việt"
    language_prop = props.get("Language", {})
    if language_prop.get("rich_text") and len(language_prop["rich_text"]) > 0:
        ngon_ngu = language_prop["rich_text"][0].get("plain_text", "Tiếng Việt")
    elif language_prop.get("select"):
        ngon_ngu = language_prop["select"].get("name", "Tiếng Việt")
    
    # Trích xuất trạng thái phê duyệt
    phe_duyet = 1 if props.get("Approved", {}).get("checkbox", False) else 0
    
    # Trích xuất ngày cập nhật (xử lý nếu date là None)
    ngay_cap_nhat = datetime.now().strftime("%Y-%m-%d")
    last_updated_prop = props.get("Last Updated", {})
    date_prop = last_updated_prop.get("date")
    if date_prop is not None:
        ngay_cap_nhat = date_prop.get("start", ngay_cap_nhat)
    else:
        logger.warning(f"Dòng {notion_id} thiếu hoặc có date None trong Last Updated.")
    
    return notion_id, cau_hoi, cau_tra_loi, danh_muc, ngon_ngu, phe_duyet, ngay_cap_nhat

# --- Đồng bộ dữ liệu vào SQLite ---
def dong_bo_sqlite(du_lieu):
    conn = khoi_tao_db()
    cur = conn.cursor()
    
    try:
        for row in du_lieu:
            try:
                notion_id, cau_hoi, cau_tra_loi, danh_muc, ngon_ngu, phe_duyet, ngay_cap_nhat = trich_xuat_thuoc_tinh(row)
                
                # Bỏ qua nếu thiếu ID, câu hỏi hoặc câu trả lời
                if not notion_id or not cau_hoi or not cau_tra_loi:
                    logger.warning(f"Bỏ qua dòng thiếu ID, câu hỏi hoặc câu trả lời: {notion_id}")
                    continue
                
                # Thực hiện upsert
                cur.execute("""
                    INSERT OR REPLACE INTO faq (id, question, answer, category, language, approved, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (notion_id, cau_hoi, cau_tra_loi, danh_muc, ngon_ngu, phe_duyet, ngay_cap_nhat))
                
                logger.info(f"Đã xử lý dòng với ID: {notion_id}")
                
            except Exception as e:
                logger.error(f"Lỗi xử lý dòng {row.get('id', 'không xác định')}: {e}")
                continue
        
        conn.commit()
        logger.info("Đồng bộ dữ liệu từ Notion sang SQLite thành công!")
    except sqlite3.Error as e:
        logger.error(f"Lỗi SQLite khi đồng bộ: {e}")
        raise
    finally:
        conn.close()

# --- Thực thi chính ---
if __name__ == "__main__":
    try:
        du_lieu = lay_du_lieu_notion()
        dong_bo_sqlite(du_lieu)
    except Exception as e:
        logger.error(f"Đồng bộ thất bại: {e}")
        print(f"Lỗi: {e}. Vui lòng kiểm tra NOTION_TOKEN, DATABASE_ID và quyền truy cập.")

