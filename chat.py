# ============================================
#  CHATBOT 4-BƯỚC – HIỂU NGHĨA, KHÔNG BỊA
#  Router (LLM + Embedding) → Rewrite (LLM)
#  → Search (Embedding + LLM Rerank) → Strict Answer (LLM)
#  Model LLM:  Groq (Split Strategy: 8B & 70B)
#  Model Emb:  BAAI/bge-m3
# ============================================

import os
import re
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import time
import random
from dotenv import load_dotenv

# ==== CẤU HÌNH GROQ (SPLIT MODEL STRATEGY) ====
GROQ_MODEL_SMART = "llama-3.3-70b-versatile"  # Dùng cho Rerank, Answer (Thông minh)
GROQ_MODEL_FAST = "llama-3.1-8b-instant"      # Dùng cho Router, Rewrite (Tốc độ)
GROQ_API_KEY = "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"

FAQ_DB_PATH = r"D:\HTML\a - Copy\faq.db"
ENV_PATH = r"D:\HTML\a - Copy\rag\.env"

# Load .env
try:
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH, override=True)
    else:
        load_dotenv()
except Exception:
    pass

if not GROQ_API_KEY:
    print("⚠ Chưa có GROQ_API_KEY.")
else:
    print(f"✅ Đã cấu hình Groq (Smart: 70B | Fast: 8B).")

FALLBACK_MSG = "Hiện tại thư viện chưa có thông tin chính xác cho câu này. Bạn mô tả rõ hơn giúp mình nhé."

# ============================================
#  EMBEDDING MODEL
# ============================================
print("Đang tải model embedding (lần đầu sẽ hơi lâu)...")
try:
    embed_model = SentenceTransformer("BAAI/bge-m3")
except Exception as e:
    print(f"⚠ Lỗi load model embedding: {e}")
    print("Đang dùng fallback model (keepitreal/vietnamese-sbert)...")
    embed_model = SentenceTransformer("keepitreal/vietnamese-sbert")


# ============================================
#  TEXT NORMALIZE
# ============================================
def normalize(x: str) -> str:
    return " ".join(x.lower().strip().split())


# ============================================
#  LLM CALL (GROQ DIRECT)
# ============================================
def llm(prompt: str, temp: float = 0.15, n: int = 1024, model: str = GROQ_MODEL_SMART) -> str:
    """
    Gọi Groq API trực tiếp với cơ chế RETRY ĐƠN GIẢN (Linear Backoff).
    Hỗ trợ chọn Model (Fast vs Smart).
    """
    if not GROQ_API_KEY:
        return ""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temp,
        "max_tokens": n,
    }

    max_retries = 3
    fixed_delay = 2.0  # Chờ cố định 2 giây nếu lỗi

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15
            )
            
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            
            if resp.status_code == 429:
                print(f"⚠ Groq quá tải (429). Đang chờ {fixed_delay}s để thử lại ({attempt+1}/{max_retries})...")
                time.sleep(fixed_delay)
                continue
                
            print(f"⚠ Lỗi Groq {resp.status_code}: {resp.text}")
            return ""

        except Exception as e:
            print(f"⚠ Lỗi gọi Groq: {e}")
            return ""
    
    print("❌ Đã thử lại 3 lần nhưng Groq vẫn bận.")
    return ""


# ============================================
#  LOAD & EMBED DB
# ============================================
print("Đang tải dữ liệu từ faq.db...")

if not os.path.exists(FAQ_DB_PATH):
    print(f"❌ Không tìm thấy file {FAQ_DB_PATH}. Hãy chạy sync_all.py / sync_faq.py trước!")
    FAQ_TEXTS, BOOK_TEXTS, MAJOR_TEXTS = [], [], []
    FAQ_EMB = np.zeros((0, 768))
    BOOK_EMB = np.zeros((0, 768))
    MAJOR_EMB = np.zeros((0, 768))
    faq_rows, book_rows, major_rows = [], [], []
else:
    conn = sqlite3.connect(FAQ_DB_PATH)
    cur = conn.cursor()

    # FAQ
    cur.execute("SELECT question, answer, category FROM faq WHERE approved = 1 OR approved IS NULL")
    faq_rows = cur.fetchall()
    FAQ_TEXTS = [normalize(f"{cat or ''}: {a or ''}") for _, a, cat in faq_rows]

    # BOOKS
    cur.execute("""
        SELECT b.name, b.author, b.year, b.quantity, b.status, m.name
        FROM books b LEFT JOIN majors m ON b.major_id = m.major_id
    """)
    book_rows = cur.fetchall()
    BOOK_TEXTS = [normalize(f"sách {n}. tác giả {a}. ngành {m or ''}") for n, a, _, _, _, m in book_rows]

    # MAJORS
    cur.execute("SELECT name, major_id, description FROM majors")
    major_rows = cur.fetchall()
    MAJOR_TEXTS = [normalize(f"ngành {n}. mã {mid}. {desc or ''}") for n, mid, desc in major_rows]

    conn.close()

    print("Đang tạo embedding (lần đầu sẽ hơi lâu)...")
    FAQ_EMB = embed_model.encode(FAQ_TEXTS, normalize_embeddings=True) if FAQ_TEXTS else np.zeros((0, 768))
    BOOK_EMB = embed_model.encode(BOOK_TEXTS, normalize_embeddings=True) if BOOK_TEXTS else np.zeros((0, 768))
    MAJOR_EMB = embed_model.encode(MAJOR_TEXTS, normalize_embeddings=True) if MAJOR_TEXTS else np.zeros((0, 768))

    print(f"✅ Đã tải: FAQ={len(faq_rows)} | BOOKS={len(book_rows)} | MAJORS={len(major_rows)}")


# ============================================
#  ROUTER – FALLBACK BẰNG EMBEDDING
# ============================================
def auto_route_by_embedding(q_vec: np.ndarray) -> str:
    best_type = "FAQ"
    best_score = -1.0

    if len(FAQ_EMB) > 0:
        s = float(np.max(np.dot(FAQ_EMB, q_vec)))
        best_type, best_score = "FAQ", s

    if len(BOOK_EMB) > 0:
        s = float(np.max(np.dot(BOOK_EMB, q_vec)))
        if s > best_score:
            best_type, best_score = "BOOKS", s

    if len(MAJOR_EMB) > 0:
        s = float(np.max(np.dot(MAJOR_EMB, q_vec)))
        if s > best_score:
            best_type, best_score = "MAJORS", s

    return best_type


def is_greeting(text: str) -> bool:
    t = text.lower().strip()
    greet_words = ["xin chào", "chào bạn", "chào ad", "hello", "hi", "alo"]
    return any(w in t for w in greet_words)


# ============================================
# 1) ROUTER – DÙNG MODEL FAST (8B)
# ============================================
def route_llm(question: str, q_vec: np.ndarray) -> str:
    if is_greeting(question) and len(question.split()) <= 4:
        print("[ROUTER] Detected GREETING")
        return "GREETING"

    prompt = f"""
Phân loại câu hỏi vào 1 trong 3 nhóm dựa trên BẢN CHẤT:

1. BOOKS (Sách & Tài liệu):
   - Chỉ chọn khi người dùng tìm kiếm TÀI LIỆU, SÁCH, GIÁO TRÌNH, LUẬN VĂN cụ thể.
   - Ví dụ: "Tìm sách Python", "Giáo trình Kinh tế lượng", "Tài liệu về AI".

2. MAJORS (Ngành học & Đào tạo):
   - Chỉ chọn khi người dùng hỏi về CHƯƠNG TRÌNH ĐÀO TẠO, TUYỂN SINH, KHOA/VIỆN.
   - Ví dụ: "Ngành CNTT học gì", "Mã ngành 7480201", "Khoa Luật ở đâu".

3. FAQ (Thông tin chung & Khác):
   - TẤT CẢ các câu hỏi còn lại.
   - Bao gồm: Quy định, Thủ tục, Giờ làm việc, Wifi, Tài khoản.
   - Bao gồm: CƠ SỞ VẬT CHẤT, ĐỊA ĐIỂM (Phòng ốc, Canteen, Bãi xe...), SỰ KIỆN.
   - Bao gồm: SỐ LƯỢNG, THỐNG KÊ (Tổng số sách, Có bao nhiêu tài liệu...).

LƯU Ý ƯU TIÊN:
- Hỏi về "Tổng số lượng", "Thống kê", "Có bao nhiêu" -> CHỌN FAQ (kể cả có từ "sách").
- Hỏi về "Ở đâu", "Phòng nào", "Tầng mấy" (Vị trí) -> CHỌN FAQ (kể cả có từ "sách").
- Nếu câu hỏi không rõ ràng -> CHỌN FAQ.

Câu hỏi: "{question}"

Chỉ trả về đúng 1 từ: FAQ hoặc BOOKS hoặc MAJORS.
"""
    # DÙNG MODEL FAST (8B)
    out = llm(prompt, temp=0.05, n=10, model=GROQ_MODEL_FAST).upper().strip()
    clean_out = re.sub(r'[^A-Z]', '', out)
    print(f"[ROUTER LLM] Output: '{out}' -> Clean: '{clean_out}'")

    if clean_out in ["FAQ", "BOOKS", "MAJORS"]:
        print(f"[ROUTER] ✅ LLM chọn: {clean_out}")
        return clean_out

    print(f"[ROUTER] ⚠️ LLM không chắc chắn -> Dùng auto_route_by_embedding (Real DB)...")
    return auto_route_by_embedding(q_vec)


# ============================================
# 2) REWRITE – DÙNG MODEL FAST (8B)
# ============================================
def rewrite_question(q: str) -> str:
    if len(q.split()) < 2:
        return q

    prompt = f"""
Bạn là một trợ lý thông minh. Hãy ĐỌC HIỂU ý định của người dùng và viết lại câu hỏi sao cho rõ ràng, đầy đủ nghĩa nhất.
Nếu câu hỏi quá ngắn, dùng từ đa nghĩa hoặc thiếu chủ ngữ, hãy diễn giải lại theo cách người bình thường sẽ hỏi đầy đủ.
ĐẶC BIỆT:
- Nếu hỏi về "số", "gọi", "alo" -> Thêm từ khóa "số điện thoại liên hệ hotline".
- Nếu hỏi về "ở đâu", "chỗ nào" -> Thêm từ khóa "địa điểm vị trí".

Ví dụ:
- "số nào" -> "số điện thoại liên hệ hotline là gì"
- "mở cửa ko" -> "giờ mở cửa hoạt động như thế nào"
- "liên hệ sao" -> "cách thức liên hệ với thư viện"

Câu gốc: "{q}"

Câu viết lại (chỉ viết 1 câu duy nhất):
"""
    # DÙNG MODEL FAST (8B)
    out = llm(prompt, temp=0.1, n=64, model=GROQ_MODEL_FAST)
    return out.strip() if out else q


# ============================================
# 3A) SEMANTIC SEARCH CHO FAQ
# ============================================
def search_faq_candidates(q_vec: np.ndarray, top_k: int = 10, filter_category: str = None):
    if len(FAQ_EMB) == 0: return []
    sims = np.dot(FAQ_EMB, q_vec)
    idx = np.argsort(-sims)[:top_k]
    candidates = []
    for i in idx:
        score = float(sims[i])
        if score < 0.08: continue
        q, a, cat = faq_rows[i]
        if filter_category and filter_category not in ["FAQ", "BOOKS", "MAJORS", "GREETING"]:
            if cat != filter_category: continue
        candidates.append({"score": score, "question": q or "", "answer": a or "", "category": cat or "", "id": i})
    return candidates


# ============================================
# 3B) SEMANTIC SEARCH CHO BOOKS / MAJORS
# ============================================
def search_nonfaq(table: str, q_vec: np.ndarray, top_k: int = 10):
    candidates = []
    if table == "BOOKS":
        if len(BOOK_EMB) == 0: return []
        sims = np.dot(BOOK_EMB, q_vec)
        rows = book_rows
        th = 0.15
        idx = np.argsort(-sims)[:top_k]
        for i in idx:
            score = float(sims[i])
            if score < th: continue
            n, a, y, qty, s, m = rows[i]
            content = f"Sách: {n}. Tác giả: {a}. Năm: {y}. Số lượng: {qty}. Tình trạng: {s}. Ngành: {m or 'Chung'}"
            candidates.append({"score": score, "question": "", "answer": content, "category": "BOOKS", "id": i})
        return candidates

    if len(MAJOR_EMB) == 0: return []
    sims = np.dot(MAJOR_EMB, q_vec)
    rows = major_rows
    th = 0.20
    idx = np.argsort(-sims)[:top_k]
    for i in idx:
        score = float(sims[i])
        if score < th: continue
        name, code, desc = rows[i]
        content = f"Ngành: {name}. Mã ngành: {code}. Mô tả: {desc or 'Đang cập nhật'}"
        candidates.append({"score": score, "question": "", "answer": content, "category": "MAJORS", "id": i})
    return candidates


# ============================================
# 3C) LLM RERANK – DÙNG MODEL SMART (70B)
# ============================================
def rerank_with_llm(user_q: str, candidates: list):
    if not candidates: return None
    block = ""
    for i, c in enumerate(candidates, start=1):
        block += f"{i}. [{c['category']}] {c['answer']}\n"

    prompt = f"""
Bạn là chuyên gia tư vấn thông minh.
Nhiệm vụ: Tìm câu trả lời PHÙ HỢP NHẤT cho câu hỏi của người dùng trong danh sách bên dưới.

Câu hỏi: "{user_q}"

Danh sách ứng viên:
{block}

HƯỚNG DẪN TƯ DUY:
- Hãy hiểu Ý NGHĨA của câu hỏi (không chỉ bắt từ khóa).
- Nếu câu hỏi tìm "Địa điểm" (ở đâu), hãy chọn câu chứa thông tin vị trí.
- Nếu câu hỏi tìm "Danh sách" (gồm những gì), hãy chọn câu liệt kê đầy đủ nhất.

YÊU CẦU:
- Nếu tìm thấy câu trả lời phù hợp: Trả về SỐ THỨ TỰ (ví dụ: 1, 2...).
- Nếu không có câu nào khớp: Trả về 0.

Chỉ trả về 1 con số duy nhất.
"""
    # DÙNG MODEL SMART (70B)
    out = llm(prompt, temp=0.1, n=128, model=GROQ_MODEL_SMART).strip()
    match = re.search(r'\d+', out)
    if match:
        idx = int(match.group()) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]

    if candidates and candidates[0]['score'] > 0.45:
        print(f"[Rerank] LLM từ chối, nhưng Top 1 score cao ({candidates[0]['score']:.2f}) -> Chọn Top 1.")
        return candidates[0]
    return None


# ============================================
# 4) STRICT ANSWER – DÙNG MODEL SMART (70B)
# ============================================
def strict_answer(question: str, knowledge: str) -> str:
    print(f"[DEBUG STRICT] Q: {question} | Knowledge: {knowledge[:50]}...")
    prompt = f"""
Bạn là trợ lý ảo của thư viện. 
NHIỆM VỤ: Trả lời câu hỏi dựa trên thông tin cung cấp bên dưới.

THÔNG TIN (KNOWLEDGE):
{knowledge}

CÂU HỎI (QUESTION): "{question}"

QUY TẮC:
1. Trả lời ngắn gọn, đúng trọng tâm bằng Tiếng Việt.
2. Dùng thông tin trong phần KNOWLEDGE để trả lời.
3. Nếu thông tin có chứa số liệu, địa điểm, quy trình -> Hãy trích xuất ra để trả lời.
4. Nếu thông tin không khớp hoàn toàn nhưng có liên quan -> Hãy trả lời dựa trên những gì có thể.

Nếu thông tin HOÀN TOÀN KHÔNG LIÊN QUAN thì mới nói: "{FALLBACK_MSG}"

Câu trả lời của bạn:
"""
    # DÙNG MODEL SMART (70B)
    out = llm(prompt, temp=0.05, n=256, model=GROQ_MODEL_SMART)
    print(f"[DEBUG STRICT OUT] {out}")

    if not out: return FALLBACK_MSG
    out = out.strip()
    if any(c.isdigit() for c in out) or "@" in out or "http" in out: return out
    if "không có thông tin" in out.lower() and len(out) < 15: return FALLBACK_MSG
    return out


# ============================================
#  MAIN PROCESS
# ============================================
def process_message(text: str) -> str:
    print("[CHAT.PY] ĐÃ GỌI NÃO")
    if not text.strip():
        return "Xin chào 👋 Bạn muốn hỏi thông tin gì trong thư viện?"

    q_vec_route = embed_model.encode(normalize(text), normalize_embeddings=True)
    route = route_llm(text, q_vec_route)

    rewritten = rewrite_question(text)
    q_vec = embed_model.encode(normalize(rewritten), normalize_embeddings=True)

    if route == "GREETING":
        return "Xin chào! Tôi là trợ lý ảo thư viện. Bạn cần tìm sách, hỏi quy định hay thông tin ngành học?"

    candidates = []
    if route == "BOOKS":
        candidates = search_nonfaq("BOOKS", q_vec, top_k=15)
    elif route == "MAJORS":
        candidates = search_nonfaq("MAJORS", q_vec, top_k=15)
    else:
        candidates = search_faq_candidates(q_vec, top_k=20)

    if not candidates:
        return "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp."

    print(f"[DEBUG {route}] Found {len(candidates)} candidates.")
    best_cand = rerank_with_llm(rewritten, candidates)
    if not best_cand:
        print("[DEBUG] ⏩ Skip Rerank -> Chọn Top 1.")
        best_cand = candidates[0]
    else:
        print(f"[DEBUG] ✅ Rerank chọn: {best_cand['answer'][:50]}...")

    return strict_answer(rewritten, best_cand['answer'])


if __name__ == "__main__":
    print("🤖 Chatbot 4-BƯỚC (Router/Rewrite: 8B | Rerank/Answer: 70B) đã sẵn sàng!")
    while True:
        q = input("\nBạn: ")
        if q.lower() in ["quit", "bye", "exit", "thoát"]:
            print("Hẹn gặp lại bạn ở thư viện nhé! 📚")
            break
        print("Bot:", process_message(q))
