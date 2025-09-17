from fastapi import FastAPI, Query
from typing import List, Optional
import sqlite3

DB_PATH = "faq.db"

app = FastAPI(title="FAQ API", description="API tra cứu FAQ từ SQLite", version="1.0")

def query_db(sql: str, params: tuple = ()):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/faq", summary="Lấy toàn bộ FAQ")
def get_all_faq():
    sql = "SELECT * FROM faq WHERE approved=1"
    return query_db(sql)

@app.get("/faq/{faq_id}", summary="Lấy FAQ theo ID")
def get_faq_by_id(faq_id: int):
    sql = "SELECT * FROM faq WHERE id=? AND approved=1"
    rows = query_db(sql, (faq_id,))
    if not rows:
        return {"message": "Không tìm thấy FAQ này."}
    return rows[0]

@app.get("/search", summary="Tìm kiếm FAQ theo từ khóa")
def search_faq(
    q: str = Query(..., description="Từ khóa tìm kiếm trong câu hỏi/trả lời"),
    category: Optional[str] = Query(None, description="Lọc theo category (tuỳ chọn)"),
    language: Optional[str] = Query(None, description="Lọc theo ngôn ngữ (tuỳ chọn, VD: 'Tiếng Việt')")
):
    sql = "SELECT * FROM faq WHERE approved=1 AND (question LIKE ? OR answer LIKE ?)"
    params = [f"%{q}%", f"%{q}%"]

    if category:
        sql += " AND category=?"
        params.append(category)
    if language:
        sql += " AND language=?"
        params.append(language)

    return query_db(sql, tuple(params))

@app.get("/", summary="Trang chào mừng")
def home():
    return {"message": "Chào mừng đến API FAQ! Hãy thử /faq hoặc /search?q=giờ"}
