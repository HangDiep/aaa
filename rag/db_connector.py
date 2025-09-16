import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


load_dotenv()
DB_URL = os.getenv("DB_URL", "sqlite:///faq.db")
engine = create_engine(DB_URL, future=True)


# Ví dụ lấy FAQ (đã duyệt)
FAQ_SQL = """
SELECT id, question, answer, category, language, last_updated
FROM faq
WHERE approved = 1
"""


def fetch_faqs():
    with engine.connect() as conn:
        rows = conn.execute(text(FAQ_SQL)).mappings().all()
        return [dict(r) for r in rows]


# Tìm kiếm gần đúng (LIKE) – có thể thay bằng FTS5 (SQLite) hoặc BM25
def search_faq_like(q: str, limit: int = 5):
    sql = text(
    """
    SELECT id, question, answer, category, language, last_updated
    FROM faq
    WHERE approved = 1 AND (question LIKE :q OR answer LIKE :q)
    LIMIT :limit
    """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"q": f"%{q}%", "limit": limit}).mappings().all()
        return [dict(r) for r in rows]