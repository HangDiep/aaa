"""
API endpoint để nhận data từ n8n và tự động update SQLite + Qdrant
Chạy: uvicorn update_api:app --host 0.0.0.0 --port 8001
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import sqlite3
import os
import subprocess
from dotenv import load_dotenv

# Load .env
ENV_PATH = r"D:\HTML\a - Copy\rag\.env"
try:
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH, override=True)
    else:
        load_dotenv()
except Exception:
    pass

FAQ_DB_PATH = os.getenv("FAQ_DB_PATH", r"D:\HTML\a - Copy\faq.db")

app = FastAPI(title="Notion Auto-Update API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
#  DATA MODELS
# ============================================
class FAQItem(BaseModel):
    question: str
    answer: str
    category: str

class BookItem(BaseModel):
    name: str
    author: str
    year: int
    quantity: int
    status: str
    major_id: str

class MajorItem(BaseModel):
    major_id: str
    name: str
    description: str

# ============================================
#  UPDATE ENDPOINTS
# ============================================
@app.post("/update/faq")
async def update_faq(items: List[FAQItem]):
    """Nhận data FAQ từ n8n và update SQLite + Qdrant"""
    try:
        conn = sqlite3.connect(FAQ_DB_PATH)
        cur = conn.cursor()
        
        # Xóa data cũ
        cur.execute("DELETE FROM faq")
        
        # Insert data mới
        for item in items:
            cur.execute(
                "INSERT INTO faq (question, answer, category, approved) VALUES (?, ?, ?, 1)",
                (item.question, item.answer, item.category)
            )
        
        conn.commit()
        conn.close()
        
        # Tự động push lên Qdrant
        subprocess.Popen(["python", "push_to_qdrant.py"], cwd=r"D:\HTML\a - Copy")
        
        return {
            "status": "success",
            "message": f"Updated {len(items)} FAQ items. Qdrant update started.",
            "count": len(items)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update/books")
async def update_books(items: List[BookItem]):
    """Nhận data BOOKS từ n8n và update SQLite + Qdrant"""
    try:
        conn = sqlite3.connect(FAQ_DB_PATH)
        cur = conn.cursor()
        
        cur.execute("DELETE FROM books")
        
        for item in items:
            cur.execute(
                "INSERT INTO books (name, author, year, quantity, status, major_id) VALUES (?, ?, ?, ?, ?, ?)",
                (item.name, item.author, item.year, item.quantity, item.status, item.major_id)
            )
        
        conn.commit()
        conn.close()
        
        subprocess.Popen(["python", "push_to_qdrant.py"], cwd=r"D:\HTML\a - Copy")
        
        return {
            "status": "success",
            "message": f"Updated {len(items)} BOOKS items. Qdrant update started.",
            "count": len(items)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update/majors")
async def update_majors(items: List[MajorItem]):
    """Nhận data MAJORS từ n8n và update SQLite + Qdrant"""
    try:
        conn = sqlite3.connect(FAQ_DB_PATH)
        cur = conn.cursor()
        
        cur.execute("DELETE FROM majors")
        
        for item in items:
            cur.execute(
                "INSERT INTO majors (major_id, name, description) VALUES (?, ?, ?)",
                (item.major_id, item.name, item.description)
            )
        
        conn.commit()
        conn.close()
        
        subprocess.Popen(["python", "push_to_qdrant.py"], cwd=r"D:\HTML\a - Copy")
        
        return {
            "status": "success",
            "message": f"Updated {len(items)} MAJORS items. Qdrant update started.",
            "count": len(items)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "message": "Notion Auto-Update API",
        "endpoints": [
            "POST /update/faq",
            "POST /update/books",
            "POST /update/majors"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
