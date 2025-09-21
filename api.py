from fastapi import FastAPI
import sqlite3
import os

app = FastAPI()

FAQ_DB_PATH = os.path.normpath(r"C:\Users\ADMIN\OneDrive\Desktop\aaa\faq.db")

# =========================
# FAQ Search
# =========================
def search_faq(query: str) -> list:
    conn = sqlite3.connect(FAQ_DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT question, answer FROM faq WHERE question LIKE ? AND approved = 1",
        (f"%{query}%",)
    )
    results = [{"question": row[0], "answer": row[1]} for row in cur.fetchall()]
    conn.close()
    return results if results else [{"answer": "Không tìm thấy câu trả lời phù hợp."}]


@app.get("/search")
async def search_faq_endpoint(q: str):
    results = search_faq(q)
    return results[0] if results else {"answer": "Không tìm thấy câu trả lời phù hợp."}


# =========================
# BOOKS Search
# =========================
def search_books(book_name_or_major: str) -> list:
    if not book_name_or_major:
        return []

    keywords = book_name_or_major.strip().lower()

    conn = sqlite3.connect(FAQ_DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ten_sach, tac_gia, nam_xuat_ban, so_luong_ton_kho, nganh, id_nganh, trang_thai
        FROM books
        WHERE approved = 1 AND (
            lower(trim(ten_sach)) LIKE '%' || ? || '%' OR
            lower(trim(nganh)) LIKE '%' || ? || '%'
        )
        """,
        (keywords, keywords)
    )
    results = []
    for row in cur.fetchall():
        results.append({
            "name": row[0],
            "author": row[1],
            "year": row[2],
            "quantity": row[3],
            "major": row[4],
            "major_id": row[5],
            "status": row[6]
        })
    conn.close()
    return results if results else [{"answer": "Không tìm thấy sách phù hợp."}]

@app.get("/inventory")
async def search_books_endpoint(book_name: str):
    results = search_books(book_name)
    return results if results else [{"answer": "Không tìm thấy sách phù hợp."}]