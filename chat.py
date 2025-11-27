# ============================================
#  CHATBOT 4-BƯỚC – HIỂU NGHĨA, KHÔNG BỊA
#  Router (LLM + Embedding) → Rewrite (LLM)
#  → Search (Embedding + LLM Rerank) → Strict Answer (LLM)
#  Model LLM:  qwen2.5:3b  (ollama)
#  Model Emb:  BAAI/bge-m3
# ============================================

import os
import re
import sqlite3
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

FAQ_DB_PATH = "faq.db"
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "qwen2.5:3b"
TIMEOUT = 20

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
#  TEXT NORMALIZE – NHẸ, KHÔNG PHÁ NGHĨA
# ============================================
def normalize(x: str) -> str:
    # chỉ lower + trim, không đụng tới dấu
    return " ".join(x.lower().strip().split())


# ============================================
#  OLLAMA LLM CALL
# ============================================
def llm(prompt: str, temp: float = 0.15, n: int = 128) -> str:
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temp, "num_predict": n},
            },
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            return r.json().get("response", "").strip()
    except Exception:
        pass
    return ""


# ============================================
#  LOAD & EMBED DB
# ============================================
print("Đang tải dữ liệu từ faq.db...")

if not os.path.exists(FAQ_DB_PATH):
    print(f"❌ Không tìm thấy file {FAQ_DB_PATH}. Hãy chạy sync_all.py / sync_faq.py trước!")
    # Tạo dummy để không crash
    FAQ_TEXTS, BOOK_TEXTS, MAJOR_TEXTS = [], [], []
    FAQ_EMB = np.zeros((0, 768))
    BOOK_EMB = np.zeros((0, 768))
    MAJOR_EMB = np.zeros((0, 768))
    faq_rows, book_rows, major_rows = [], [], []
else:
    conn = sqlite3.connect(FAQ_DB_PATH)
    cur = conn.cursor()

    # FAQ
    cur.execute(
        "SELECT question, answer, category FROM faq WHERE approved = 1 OR approved IS NULL"
    )
    faq_rows = cur.fetchall()

    FAQ_TEXTS = []
    for q, a, cat in faq_rows:
        # Nhúng Category + Answer để tạo chunk kiến thức rõ nghĩa
        content = f"{cat or ''}: {a or ''}"
        FAQ_TEXTS.append(normalize(content))

    # BOOKS
    cur.execute(
        """
        SELECT b.name, b.author, b.year, b.quantity, b.status, m.name
        FROM books b LEFT JOIN majors m ON b.major_id = m.major_id
        """
    )
    book_rows = cur.fetchall()
    BOOK_TEXTS = [
        normalize(f"sách {n}. tác giả {a}. ngành {m or ''}")
        for n, a, _, _, _, m in book_rows
    ]

    # MAJORS
    cur.execute("SELECT name, major_id, description FROM majors")
    major_rows = cur.fetchall()
    MAJOR_TEXTS = [
        normalize(f"ngành {n}. mã {mid}. {desc or ''}")
        for n, mid, desc in major_rows
    ]

    conn.close()

    print("Đang tạo embedding (lần đầu sẽ hơi lâu)...")
    FAQ_EMB = (
        embed_model.encode(FAQ_TEXTS, normalize_embeddings=True)
        if FAQ_TEXTS
        else np.zeros((0, 768))
    )
    BOOK_EMB = (
        embed_model.encode(BOOK_TEXTS, normalize_embeddings=True)
        if BOOK_TEXTS
        else np.zeros((0, 768))
    )
    MAJOR_EMB = (
        embed_model.encode(MAJOR_TEXTS, normalize_embeddings=True)
        if MAJOR_TEXTS
        else np.zeros((0, 768))
    )

    print(f"✅ Đã tải: FAQ={len(faq_rows)} | BOOKS={len(book_rows)} | MAJORS={len(major_rows)}")


# ============================================
#  ROUTER – FALLBACK BẰNG EMBEDDING (REAL DB)
# ============================================
def auto_route_by_embedding(q_vec: np.ndarray) -> str:
    """
    Nếu LLM phân loại linh tinh → dùng embedding chọn bảng nào gần nhất
    dựa trên dữ liệu thật trong FAQ/BOOKS/MAJORS.
    """
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


# ============================================
#  SIMPLE GREETING CHECK
# ============================================
def is_greeting(text: str) -> bool:
    t = text.lower().strip()
    greet_words = ["xin chào", "chào bạn", "chào ad", "hello", "hi", "alo"]
    return any(w in t for w in greet_words)


# ============================================
# 1) ROUTER – 100% LLM + EMBEDDING (KHÔNG DÙNG data.pth)
# ============================================
def route_llm(question: str, q_vec: np.ndarray) -> str:
    """
    HYBRID ROUTER:
    1. Hỏi LLM (Reasoning): "Câu này thuộc nhóm nào?"
    2. Nếu LLM trả đúng (BOOKS/MAJORS/FAQ) -> Tin nó.
    3. Nếu LLM trả linh tinh -> Dùng auto_route_by_embedding (vector từ DB thật).
    """
    # B0: Check Greeting nhanh
    if is_greeting(question) and len(question.split()) <= 4:
        print("[ROUTER] Detected GREETING")
        return "GREETING"

    # B1: Dùng LLM (Reasoning)
    prompt = f"""
Phân loại câu hỏi vào 1 trong 3 nhóm:
1. FAQ: Quy định, thủ tục, giờ mở cửa, liên hệ, wifi, TỔNG SỐ LƯỢNG tài liệu, thống kê, VỊ TRÍ phòng ốc, địa điểm...
2. BOOKS: Tìm sách cụ thể, giáo trình, tài liệu tham khảo, tác giả, kiểm tra sách còn không...
3. MAJORS: Ngành học, mã ngành, chương trình đào tạo, khoa...

LƯU Ý: 
- Hỏi về "Tổng số lượng" hoặc "Thống kê" -> Chọn FAQ.
- Hỏi về "Ở đâu", "Phòng nào", "Tầng mấy" -> Chọn FAQ.
- Hỏi về "Quy trình", "Thủ tục", "Cách mượn/trả" -> Chọn FAQ (kể cả có từ "sách").

Câu hỏi: "{question}"

Chỉ trả về đúng 1 từ: FAQ hoặc BOOKS hoặc MAJORS.
"""
    out = llm(prompt, temp=0.05, n=10)
    out_upper = (out or "").upper()
    print(f"[ROUTER LLM] Raw output: {out_upper!r}")

    if "FAQ" in out_upper:
        print("[ROUTER] ✅ LLM chọn: FAQ")
        return "FAQ"
    if "BOOKS" in out_upper:
        print("[ROUTER] ✅ LLM chọn: BOOKS")
        return "BOOKS"
    if "MAJORS" in out_upper:
        print("[ROUTER] ✅ LLM chọn: MAJORS")
        return "MAJORS"

    # B2: Fallback = Vector theo DB thật
    print("[ROUTER] ⚠️ LLM không chắc chắn -> Dùng auto_route_by_embedding (Real DB)...")
    fallback_route = auto_route_by_embedding(q_vec)
    print(f"[ROUTER] -> Vector (DB) chọn: {fallback_route}")
    return fallback_route


# ============================================
# 2) REWRITE – KHÔNG ĐỤNG CÂU QUÁ NGẮN
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
    out = llm(prompt, temp=0.1, n=64)
    return out.strip() if out else q


# ============================================
# 3A) SEMANTIC SEARCH CHO FAQ
# ============================================
def search_faq_candidates(q_vec: np.ndarray, top_k: int = 10, filter_category: str = None):
    if len(FAQ_EMB) == 0:
        return []

    sims = np.dot(FAQ_EMB, q_vec)
    idx = np.argsort(-sims)[:top_k]

    candidates = []
    for i in idx:
        score = float(sims[i])
        if score < 0.08:
            continue

        q, a, cat = faq_rows[i]

        if filter_category and filter_category not in ["FAQ", "BOOKS", "MAJORS", "GREETING"]:
            if cat != filter_category:
                continue

        candidates.append(
            {
                "score": score,
                "question": q or "",
                "answer": a or "",
                "category": cat or "",
                "id": i,
            }
        )
    return candidates


# ============================================
# 3B) SEMANTIC SEARCH CHO BOOKS / MAJORS
# ============================================
def search_nonfaq(table: str, q_vec: np.ndarray, top_k: int = 10):
    candidates = []

    if table == "BOOKS":
        if len(BOOK_EMB) == 0:
            return []
        sims = np.dot(BOOK_EMB, q_vec)
        rows = book_rows
        th = 0.15
        idx = np.argsort(-sims)[:top_k]
        for i in idx:
            score = float(sims[i])
            if score < th:
                continue
            n, a, y, qty, s, m = rows[i]
            content = (
                f"Sách: {n}. Tác giả: {a}. Năm: {y}. "
                f"Số lượng: {qty}. Tình trạng: {s}. Ngành: {m or 'Chung'}"
            )
            candidates.append({
                "score": score,
                "question": "",
                "answer": content,
                "category": "BOOKS",
                "id": i
            })
        return candidates

    # MAJORS
    if len(MAJOR_EMB) == 0:
        return []
    sims = np.dot(MAJOR_EMB, q_vec)
    rows = major_rows
    th = 0.20
    idx = np.argsort(-sims)[:top_k]
    for i in idx:
        score = float(sims[i])
        if score < th:
            continue
        name, code, desc = rows[i]
        content = f"Ngành: {name}. Mã ngành: {code}. Mô tả: {desc or 'Đang cập nhật'}"
        candidates.append({
            "score": score,
            "question": "",
            "answer": content,
            "category": "MAJORS",
            "id": i
        })
    return candidates


# ============================================
# 3C) LLM RERANK CHO FAQ/BOOKS/MAJORS
# ============================================
def rerank_with_llm(user_q: str, candidates: list):
    if not candidates:
        return None

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
- Ví dụ: Hỏi "Fanpage" thì câu chứa "Facebook" là đúng. Hỏi "Quy trình" thì câu hướng dẫn các bước là đúng.
- Nếu câu hỏi tìm "Địa điểm" (ở đâu), hãy chọn câu chứa thông tin vị trí.
- Nếu câu hỏi tìm "Danh sách" (gồm những gì), hãy chọn câu liệt kê đầy đủ nhất.

YÊU CẦU:
- Nếu tìm thấy câu trả lời phù hợp: Trả về SỐ THỨ TỰ (ví dụ: 1, 2...).
- Nếu không có câu nào khớp: Trả về 0.

Chỉ trả về 1 con số duy nhất.
"""
    out = llm(prompt, temp=0.1, n=128).strip()

    match = re.search(r'\d+', out)
    if match:
        idx = int(match.group()) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]

    # Fallback: tin top 1 nếu score rất cao
    if candidates and candidates[0]['score'] > 0.45:
        print(f"[Rerank] LLM từ chối, nhưng Top 1 score cao ({candidates[0]['score']:.2f}) -> Chọn Top 1.")
        return candidates[0]

    return None


def strict_answer(question: str, knowledge: str) -> str:
    print(f"[DEBUG STRICT] Q: {question} | Knowledge: {knowledge[:50]}...")
    prompt = f"""
Bạn là trợ lý ảo của thư viện. 
NHIỆM VỤ: Trả lời câu hỏi dựa trên thông tin cung cấp bên dưới.

THÔNG TIN (KNOWLEDGE):
{knowledge}

CÂU HỎI (QUESTION): "{question}"

QUY TẮC BẮT BUỘC:
1. ⚠️ TUYỆT ĐỐI TRẢ LỜI BẰNG TIẾNG VIỆT.
2. Nếu thông tin có vẻ liên quan (dù chỉ một phần), HÃY TRẢ LỜI NGAY.
3. Ví dụ: Hỏi "sách công nghệ" mà có "Công nghệ phần mềm" -> TRẢ LỜI thông tin sách đó.
4. Nếu thông tin là danh sách, hãy trích xuất ý chính.
5. ⚠️ ĐỐI VỚI TÊN RIÊNG (Tác giả, Tên sách, Người liên hệ...): PHẢI TRÍCH XUẤT CHÍNH XÁC 100%, KHÔNG ĐƯỢC RÚT GỌN.
6. Nếu câu hỏi dùng từ đồng nghĩa, hãy tự suy luận.
7. Nếu có số liệu/thống kê, hãy đưa ra con số đó.
8. Tuyệt đối KHÔNG trả lời "{FALLBACK_MSG}" nếu bạn tìm thấy thông tin liên quan.

Nếu thông tin HOÀN TOÀN KHÔNG LIÊN QUAN thì mới nói: "{FALLBACK_MSG}"

Câu trả lời của bạn (Tiếng Việt):
"""
    out = llm(prompt, temp=0.05, n=256)
    print(f"[DEBUG STRICT OUT] {out}")

    if not out:
        return FALLBACK_MSG

    out = out.strip()

    # Chấp nhận câu trả lời có số / email / link
    if any(c.isdigit() for c in out) or "@" in out or "http" in out:
        return out

    if "không có thông tin" in out.lower() and len(out) < 15:
        return FALLBACK_MSG

    return out


# ============================================
#  MAIN PROCESS
# ============================================
def process_message(text: str) -> str:
    print("[CHAT.PY] ĐÃ GỌI NÃO")
    if not text.strip():
        return "Xin chào 👋 Bạn muốn hỏi thông tin gì trong thư viện?"

    # B0: vector cho router
    q_vec_route = embed_model.encode(normalize(text), normalize_embeddings=True)

    # B1: Router (LLM + Embedding)
    route = route_llm(text, q_vec_route)

    # B2: Rewrite
    rewritten = rewrite_question(text)
    q_vec = embed_model.encode(normalize(rewritten), normalize_embeddings=True)


    if route == "GREETING":
        return "Xin chào! Tôi là trợ lý ảo thư viện. Bạn cần tìm sách, hỏi quy định hay thông tin ngành học?"

    # BOOKS
    if route == "BOOKS":
        candidates = search_nonfaq("BOOKS", q_vec, top_k=15)
        if not candidates:
            return "Không tìm thấy sách nào phù hợp."

        print(f"[DEBUG BOOKS] Found {len(candidates)} candidates.")
        best_cand = rerank_with_llm(rewritten, candidates)
        if not best_cand:
            best_cand = candidates[0]

        return strict_answer(rewritten, best_cand['answer'])

    # MAJORS
    if route == "MAJORS":
        candidates = search_nonfaq("MAJORS", q_vec, top_k=15)
        if not candidates:
            return "Không tìm thấy ngành học nào phù hợp."

        print(f"[DEBUG MAJORS] Found {len(candidates)} candidates.")
        best_cand = rerank_with_llm(rewritten, candidates)
        if not best_cand:
            best_cand = candidates[0]

        return strict_answer(rewritten, best_cand['answer'])

    # Mặc định: FAQ
    filter_cat = None  # hiện tại chưa lọc theo category nhỏ
    print(f"\n[DEBUG] Filter Category: {filter_cat}")

    candidates = search_faq_candidates(q_vec, top_k=20, filter_category=None)

    if not candidates:
        print("[DEBUG] ❌ Không tìm thấy candidate nào (do điểm thấp hơn ngưỡng).")
        return "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp trong cơ sở dữ liệu."

    print(f"[DEBUG] Found {len(candidates)} candidates:")
    for c in candidates:
        print(f"  - [{c['score']:.4f}] {c['answer'][:50]}... (Cat: {c['category']})")

    best_cand = rerank_with_llm(rewritten, candidates)
    if not best_cand:
        print("[DEBUG] ❌ Rerank LLM từ chối tất cả candidates. Lấy Top 1.")
        best_cand = candidates[0]
    else:
        print(f"[DEBUG] ✅ Rerank chọn: {best_cand['answer'][:50]}...")

    final_ans = strict_answer(rewritten, best_cand['answer'])
    return final_ans


# ============================================
#  CLI
# ============================================
if __name__ == "__main__":
    print("🤖 Chatbot 4-BƯỚC (Router → Rewrite → Search+Rerank → Strict Answer) đã sẵn sàng!")
    while True:
        q = input("\nBạn: ")
        if q.lower() in ["quit", "bye", "exit", "thoát"]:
            print("Hẹn gặp lại bạn ở thư viện nhé! 📚")
            break
        print("Bot:", process_message(q))
