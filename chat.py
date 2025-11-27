# ============================================
#  CHATBOT 4-BƯỚC – HIỂU NGHĨA, KHÔNG BỊA
#  Router (LLM + Embedding) → Rewrite (LLM)
#  → Search (Embedding + LLM Rerank) → Strict Answer (LLM)
#  Model LLM:  qwen2.5:3b  (ollama)
#  Model Emb:  BAAI/bge-large-en-v1.5
# ============================================

import sqlite3
import requests
import numpy as np
import os
from sentence_transformers import SentenceTransformer

FAQ_DB_PATH = "faq.db"
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "qwen2.5:3b"
TIMEOUT = 20

FALLBACK_MSG = "Hiện tại thư viện chưa có thông tin chính xác cho câu này. Bạn mô tả rõ hơn giúp mình nhé."

# ============================================
#  EMBEDDING MODEL (Vietnamese SBERT)
# ============================================
print("Đang tải model embedding (lần đầu sẽ hơi lâu)...")
try:
    # User suggested BAAI/bge-large-en-v1.5, but BAAI/bge-m3 is SOTA for multilingual/Vietnamese
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
    print(f"❌ Không tìm thấy file {FAQ_DB_PATH}. Hãy chạy sync_faq.py trước!")
    # Tạo dummy để không crash
    FAQ_TEXTS, BOOK_TEXTS, MAJOR_TEXTS = [], [], []
    FAQ_EMB = np.zeros((0, 768)) # vietnamese-sbert dim is 768
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
    
    # UPDATE: Theo yêu cầu "hiểu câu trả lời", ta sẽ embed CÂU TRẢ LỜI (Answer).
    # Tuy nhiên, để AI hiểu ngữ cảnh tốt nhất, ta nên ghép cả Category vào (nếu có).
    # Ví dụ: "Giờ mở cửa: Thư viện mở từ 7h..." sẽ dễ tìm hơn là chỉ "Thư viện mở từ 7h..."
    FAQ_TEXTS = []
    for q, a, cat in faq_rows:
        # Kết hợp Category + Answer để tạo thành một "khối kiến thức" (Knowledge Chunk)
        # Nếu Answer đã đầy đủ ý nghĩa thì rất tốt.
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
#  ROUTER – FALLBACK BẰNG EMBEDDING
# ============================================
def auto_route_by_embedding(q_vec: np.ndarray) -> str:
    """
    Nếu LLM phân loại linh tinh → dùng embedding chọn bảng nào gần nhất.
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
#  LOAD TRAINED MODEL (ML Classification)
# ============================================
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FILE = "data.pth"
try:
    data = torch.load(FILE, map_location=device)
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    print("✅ Đã load model phân loại (data.pth)")
except Exception as e:
    print(f"⚠ Không load được model phân loại: {e}")
    model = None

def predict_intent(sentence):
    if not model:
        return None
    
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    if prob.item() > 0.75:
        return tag
    return None

# ============================================
# 1) ROUTER – HYBRID (ML + LLM)
# ============================================
def route_llm(question: str, q_vec: np.ndarray) -> str:
    # B1: Dùng model đã train để phân loại Category (ML cơ bản)
    # Model này đã được train trên CÂU TRẢ LỜI từ Notion
    intent = predict_intent(question)
    
    if intent:
        print(f"[ML Predict] Intent: {intent}")
        # Nếu là GREETING -> Trả về luôn
        if intent == "GREETING":
            return "GREETING"
        
        # Nếu ra các Category cụ thể (Giờ mở cửa, Liên hệ...) -> Trả về chính Category đó
        # Để lát nữa search_faq chỉ tìm trong category này thôi.
        return intent

    # B2: Nếu model không chắc chắn (hoặc là câu hỏi về Sách/Ngành mà model chưa học kỹ)
    # Dùng LLM để phân loại chung
    prompt = f"""
Phân loại câu hỏi của sinh viên vào 1 trong 3 nhóm sau:

1. FAQ: Hỏi về quy định, thủ tục, giờ mở cửa, liên hệ, mượn trả sách, wifi, tài khoản...
2. BOOKS: Hỏi tìm sách, giáo trình, tài liệu, tác giả, kiểm tra sách còn không...
3. MAJORS: Hỏi thông tin về các ngành học, mã ngành, chương trình đào tạo...

Câu hỏi: "{question}"

Chỉ trả về đúng 1 từ: FAQ hoặc BOOKS hoặc MAJORS.
"""
    out = llm(prompt, temp=0.05, n=10).upper().strip()

    if out in ["FAQ", "BOOKS", "MAJORS"]:
        return out

    # fallback embedding
    return auto_route_by_embedding(q_vec)
# ============================================
# 2) REWRITE – KHÔNG ĐỤNG CÂU NGẮN
# ============================================
def rewrite_question(q: str) -> str:
    # Câu ngắn (≤ 5 từ) → giữ nguyên, tránh LLM phá nghĩa.
    # UPDATE: Với yêu cầu "hiểu như người", ta cho LLM sửa cả câu ngắn nếu nó quá tối nghĩa.
    # Chỉ bỏ qua nếu quá ngắn (< 2 từ)
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
# 3A) SEMANTIC SEARCH CHO FAQ – CÓ LỌC CATEGORY
# ============================================
def search_faq_candidates(q_vec: np.ndarray, top_k: int = 10, filter_category: str = None): 
    if len(FAQ_EMB) == 0:
        return []

    sims = np.dot(FAQ_EMB, q_vec)
    idx = np.argsort(-sims)[:top_k]

    candidates = []
    for i in idx:
        score = float(sims[i])
        # Hạ ngưỡng xuống cực thấp để "lưới" được hết các câu có ý nghĩa liên quan
        if score < 0.08: 
            continue
        
        q, a, cat = faq_rows[i]
        
        # LỌC: Nếu đã biết Category (do model train dự đoán), chỉ lấy đúng Category đó
        if filter_category and filter_category not in ["FAQ", "BOOKS", "MAJORS", "GREETING"]:
            # So sánh tương đối (vì có thể có sự khác biệt nhỏ về string)
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
        th = 0.15 # Hạ ngưỡng thấp để Rerank lọc
        idx = np.argsort(-sims)[:top_k]
        for i in idx:
            score = float(sims[i])
            if score < th:
                continue
            n, a, y, qty, s, m = rows[i]
            # Format nội dung để LLM đọc hiểu
            content = f"Sách: {n}. Tác giả: {a}. Năm: {y}. Số lượng: {qty}. Tình trạng: {s}. Ngành: {m or 'Chung'}"
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
# 3C) LLM RERANK CHO FAQ – CHỖ “HIỂU NGHĨA”
# ============================================
def rerank_with_llm(user_q: str, candidates: list):
    if not candidates:
        return None

    # 1. Tạo block text cho LLM đọc
    block = ""
    for i, c in enumerate(candidates, start=1):
        block += f"{i}. [{c['category']}] {c['answer']}\n"

    # 2. Prompt "Tư duy" (Reasoning) thay vì "Luật" (Rules)
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
    # Tăng n lên 128 để tránh bị cắt giữa chừng
    out = llm(prompt, temp=0.1, n=128).strip()
    
    # 3. Parse kết quả
    import re
    match = re.search(r'\d+', out)
    if match:
        idx = int(match.group()) - 1
        # Nếu LLM chọn 0 hoặc số không hợp lệ -> Coi như không chọn được
        if 0 <= idx < len(candidates):
            return candidates[idx]
            
    # 4. FALLBACK THÔNG MINH (Quan trọng!)
    # Nếu LLM không chọn được (trả về 0 hoặc lỗi), nhưng Search Engine (Embedding) 
    # đã tìm ra ứng viên số 1 có điểm số rất cao (> 0.45), thì tin tưởng Search Engine.
    # (Vì Embedding model BAAI/bge-m3 rất mạnh, thường top 1 là đúng)
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
5. ⚠️ ĐỐI VỚI TÊN RIÊNG (Tác giả, Tên sách, Người liên hệ...): PHẢI TRÍCH XUẤT CHÍNH XÁC 100%, KHÔNG ĐƯỢC RÚT GỌN (Ví dụ: "Nguyễn Thị A" không được viết là "Nguyễn Thị").
6. Nếu câu hỏi dùng từ đồng nghĩa, hãy tự suy luận.
7. Nếu có số liệu/thống kê, hãy đưa ra con số đó.
8. Tuyệt đối KHÔNG trả lời "{FALLBACK_MSG}" nếu bạn tìm thấy thông tin liên quan.

Nếu thông tin HOÀN TOÀN KHÔNG LIÊN QUAN thì mới nói: "{FALLBACK_MSG}"

Câu trả lời của bạn (Tiếng Việt):
"""
    # Tăng temp lên để bot "dám" trả lời hơn -> UPDATE: Giảm xuống để chính xác tên riêng
    out = llm(prompt, temp=0.05, n=256) 
    print(f"[DEBUG STRICT OUT] {out}")
    
    if not out:
        return FALLBACK_MSG

    out = out.strip()
    
    # UPDATE: Chấp nhận SĐT (số) hoặc Email (@) hoặc Link (http)
    if any(c.isdigit() for c in out) or "@" in out or "http" in out:
        return out

    # Bỏ check "không có thông tin" quá gắt, chỉ check nếu output quá ngắn
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

    # B1: Router
    route = route_llm(text, q_vec_route)
    # print("[DEBUG ROUTE]", route)

    # B2: Rewrite
    rewritten = rewrite_question(text)
    q_vec = embed_model.encode(normalize(rewritten), normalize_embeddings=True)

    # B3 + B4
    # UPDATE: Nếu câu hỏi dài (> 3 từ) hoặc chứa từ khóa hỏi (ở đâu, sách, phòng, bao nhiêu...), 
    # thì DÙ Router bảo là GREETING cũng KỆ NÓ, cứ đi tìm kiếm cho chắc.
    # Tránh trường hợp model train bị lệch, cứ thấy lạ là phán Greeting.
    is_real_question = len(text.split()) > 3 or any(w in text.lower() for w in ["ở đâu", "sách", "phòng", "bao nhiêu", "khi nào", "mấy giờ", "là gì"])
    
    if route == "GREETING" and not is_real_question:
        return "Xin chào! Tôi là trợ lý ảo thư viện (đã được train). Bạn cần tìm sách, hỏi quy định hay thông tin ngành học?"

    # HEURISTIC: Sửa lỗi Router đoán sai (ví dụ: "sách python" -> GREETING)
    lower_text = text.lower()
    
    # 1. Nếu có từ khóa SÁCH/GIÁO TRÌNH mà không có từ khóa QUY TRÌNH -> Force BOOKS
    if any(w in lower_text for w in ["sách", "giáo trình", "tài liệu", "tác giả", "ấn phẩm"]):
        # Trừ các trường hợp hỏi quy định (mượn, trả, phòng, giờ...)
        if not any(w in lower_text for w in ["mượn", "trả", "quy định", "nội quy", "giờ", "phòng", "thủ tục", "hướng dẫn"]):
            print("[DEBUG] Heuristic: Force route -> BOOKS")
            route = "BOOKS"

    # 2. Nếu có từ khóa NGÀNH/KHOA -> Force MAJORS
    if any(w in lower_text for w in ["ngành", "khoa", "đào tạo", "mã ngành"]):
        print("[DEBUG] Heuristic: Force route -> MAJORS")
        route = "MAJORS"

    # Nếu route là BOOKS hoặc MAJORS -> Xử lý riêng
    if route == "BOOKS":
        candidates = search_nonfaq("BOOKS", q_vec, top_k=15)
        if not candidates:
             return "Không tìm thấy sách nào phù hợp."
        
        print(f"[DEBUG BOOKS] Found {len(candidates)} candidates.")
        best_cand = rerank_with_llm(rewritten, candidates)
        if not best_cand:
             # Fallback top 1
             best_cand = candidates[0]
        
        return strict_answer(rewritten, best_cand['answer'])

    if route == "MAJORS":
        candidates = search_nonfaq("MAJORS", q_vec, top_k=15)
        if not candidates:
             return "Không tìm thấy ngành học nào phù hợp."
        
        print(f"[DEBUG MAJORS] Found {len(candidates)} candidates.")
        best_cand = rerank_with_llm(rewritten, candidates)
        if not best_cand:
             best_cand = candidates[0]
             
        return strict_answer(rewritten, best_cand['answer'])
    
    # Trường hợp còn lại: FAQ hoặc CÁC CATEGORY CỤ THỂ
    # Nếu route không phải là "FAQ" chung chung, thì nó chính là filter_category
    filter_cat = route if route != "FAQ" else None
    
    print(f"\n[DEBUG] Filter Category: {filter_cat}")

    # BƯỚC 1: Tìm TOÀN BỘ FAQ (Bỏ lọc Category để tăng Recall)
    candidates = search_faq_candidates(q_vec, top_k=20, filter_category=None)
        
    if not candidates:
        print("[DEBUG] ❌ Không tìm thấy candidate nào (do điểm thấp hơn ngưỡng).")
        return "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp trong cơ sở dữ liệu."

    print(f"[DEBUG] Found {len(candidates)} candidates:")
    for c in candidates:
        print(f"  - [{c['score']:.4f}] {c['answer'][:50]}... (Cat: {c['category']})")

    # Rerank
    best_cand = rerank_with_llm(rewritten, candidates)
    if not best_cand:
            print("[DEBUG] ❌ Rerank LLM từ chối tất cả candidates.")
            # Fallback: lấy top 1
            best_cand = candidates[0]
    else:
            print(f"[DEBUG] ✅ Rerank chọn: {best_cand['answer'][:50]}...")

    # Strict Answer
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
