import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv
from retriever import hybrid_search


load_dotenv()
app = FastAPI(title="Library RAG", version="1.0")


class AskIn(BaseModel):
    query: str
    top_k: int | None = None


class AnswerOut(BaseModel):
    answer: str
    citations: List[Dict]


# Trình ghép câu trả lời đơn giản (không LLM)
def compose_answer(query: str, hits: Dict[str, List[Dict]]) -> AnswerOut:
# 1) Ưu tiên FAQ trùng khớp LIKE
    faq_hits = hits.get("faq_like", [])
    if faq_hits:
        h = faq_hits[0]
        ans = h.get("answer")
        cite = [{"source": "faq", "id": h.get("id"), "title": h.get("question")}]
        return AnswerOut(answer=ans, citations=cite)


# 2) Nếu không có FAQ thì dùng semantic hits
    sem_hits = hits.get("semantic", [])
    if sem_hits:
        top = sem_hits[0]
        meta = top["meta"]
        # Demo: đáp bằng tóm tắt ngắn + link nguồn nếu có
        title = meta.get("title") or meta.get("question")
        url = meta.get("url")
        ans = f"Theo nguồn: {title}. \n(Mẹo: hỏi cụ thể hơn để mình trích thông tin chi tiết.)"
        cite = [{"source": meta.get("source"), "title": title, "url": url, "score": top["score"]}]
        return AnswerOut(answer=ans, citations=cite)


# 3) Fallback
    return AnswerOut(answer="Xin lỗi, mình chưa tìm thấy thông tin phù hợp.", citations=[])


@app.post("/ask", response_model=AnswerOut)
async def ask(inp: AskIn):
    k = inp.top_k or int(os.getenv("TOP_K", 5))
    hits = hybrid_search(inp.query, k)
    return compose_answer(inp.query, hits)


@app.get("/")
async def home():
    return {"message": "Library RAG is running. Try POST /ask"}