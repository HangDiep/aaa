"""
API nhận dữ liệu từ n8n (Notion Trigger) và ghi vào SQLite.

Mỗi lần bạn sửa 1 dòng và tick Approved trong Notion:
Notion Trigger -> n8n -> HTTP POST vào 3 endpoint dưới đây
    /notion/faq
    /notion/book
    /notion/major

Bảng trong SQLite:
    faq, books, majors
đều có khóa chính là notion_id (id của page trong Notion).
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import sqlite3
import os

DB_PATH = os.getenv("FAQ_DB_PATH", "faq.db")

app = FastAPI(title="Notion Trigger Sync API")


# ==========================
#  Pydantic models
# ==========================

class FAQItem(BaseModel):
    notion_id: str
    question: str
    answer: str
    category: Optional[str] = None
    language: Optional[str] = "vi"
    approved: Optional[int] = 1


class BookItem(BaseModel):
    notion_id: str
    name: str  # Changed from 'title'
    author: Optional[str] = None
    year: Optional[int] = None
    quantity: Optional[int] = 0
    status: Optional[str] = "Có sẵn"
    major_id: Optional[str] = None


class MajorItem(BaseModel):
    notion_id: str                  # ID page Notion – dùng làm khóa chính
    name: str                       # Tên ngành
    description: Optional[str] = None
    major_id: Optional[str] = None  # KHÔNG bắt buộc nữa

# ==========================
#  DB helper
# ==========================

def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    """Schema khớp với faq.db hiện tại"""
    conn = get_conn()
    cur = conn.cursor()

    # Bảng FAQ
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS faq (
            notion_id   TEXT PRIMARY KEY,
            question    TEXT,
            answer      TEXT,
            category    TEXT,
            language    TEXT,
            approved    INTEGER,
            last_updated TEXT
        )
        """
    )

    # Bảng Sách
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS books (
            notion_id   TEXT PRIMARY KEY,
            name        TEXT,
            author      TEXT,
            year        INTEGER,
            quantity    INTEGER,
            status      TEXT,
            last_updated TEXT,
            major_id    TEXT
        )
        """
    )

    # Bảng Ngành
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS majors (
            notion_id   TEXT PRIMARY KEY,
            major_id    TEXT,
            name        TEXT,
            description TEXT
        )
        """
    )

    conn.commit()
    conn.close()


@app.on_event("startup")
def on_startup():
    init_db()

class DeletePayload(BaseModel):
    notion_id: str
# ==========================
#  FAQ endpoint
# ==========================

@app.post("/notion/faq")
def upsert_faq(item: FAQItem):
    try:
        print(f"📥 Received FAQ data: {item.dict()}")  # Debug log
        
        conn = get_conn()
        cur = conn.cursor()
        now = datetime.utcnow().isoformat()

        cur.execute(
            """
            INSERT INTO faq (notion_id, question, answer, category, language, approved, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(notion_id) DO UPDATE SET
                question     = excluded.question,
                answer       = excluded.answer,
                category     = excluded.category,
                language     = excluded.language,
                approved     = excluded.approved,
                last_updated = excluded.last_updated
            """,
            (item.notion_id, item.question, item.answer, item.category, item.language, item.approved, now),
        )

        conn.commit()
        conn.close()
        print(f"✅ Inserted/Updated FAQ: {item.notion_id}")  # Debug log
        return {"status": "ok", "source": "faq", "notion_id": item.notion_id}
    except Exception as e:
        print(f"❌ Error: {e}")  # Debug log
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/notion/faq/delete")
@app.delete("/notion/faq/delete")
def delete_faq(payload: DeletePayload):
    """Xóa FAQ khi bỏ tích Approved"""
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM faq WHERE notion_id = ?", (payload.notion_id,))
        conn.commit()
        conn.close()
        
        return {"status": "deleted", "source": "faq", "notion_id": payload.notion_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================
#  BOOKS endpoint
# ==========================

@app.post("/notion/book")
def upsert_book(item: BookItem):
    try:
        conn = get_conn()
        cur = conn.cursor()
        now = datetime.utcnow().isoformat()

        cur.execute(
            """
            INSERT INTO books (notion_id, name, author, year, quantity, status, last_updated, major_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(notion_id) DO UPDATE SET
                name         = excluded.name,
                author       = excluded.author,
                year         = excluded.year,
                quantity     = excluded.quantity,
                status       = excluded.status,
                last_updated = excluded.last_updated,
                major_id     = excluded.major_id
            """,
            (
                item.notion_id,
                item.name,
                item.author,
                item.year,
                item.quantity,
                item.status,
                now,
                item.major_id,
            ),
        )

        conn.commit()
        conn.close()
        return {"status": "ok", "source": "book", "notion_id": item.notion_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/notion/book/delete")
@app.delete("/notion/book/delete")
def delete_book(payload: DeletePayload):
    """Xóa BOOK khi bỏ tích Approved"""
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM books WHERE notion_id = ?", (payload.notion_id,))
        conn.commit()
        conn.close()
        
        return {"status": "deleted", "source": "book", "notion_id": payload.notion_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# ==========================
#  MAJORS endpoint
# ==========================

@app.post("/notion/major")
def upsert_major(item: MajorItem):
    try:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO majors (notion_id, major_id, name, description)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(notion_id) DO UPDATE SET
                major_id    = excluded.major_id,
                name        = excluded.name,
                description = excluded.description
            """,
            (item.notion_id, item.major_id, item.name, item.description),
        )

        conn.commit()
        conn.close()
        return {"status": "ok", "source": "major", "notion_id": item.notion_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/notion/major/delete")
@app.delete("/notion/major/delete")
def delete_major(payload: DeletePayload):
    """Xóa MAJOR khi bỏ tích Approved"""
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM majors WHERE notion_id = ?", (payload.notion_id,))
        conn.commit()
        conn.close()
        
        return {"status": "deleted", "source": "major", "notion_id": payload.notion_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
