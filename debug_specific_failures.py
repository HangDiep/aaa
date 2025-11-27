import os
import torch
from sentence_transformers import SentenceTransformer, util
import sqlite3
import re

import numpy as np

import numpy as np
import requests
import re

# Load model (same as chat.py)
embed_model = SentenceTransformer('BAAI/bge-m3')
DB_PATH = 'faq.db'

# --- MOCK OLLAMA CALL ---
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "qwen2.5:3b"
TIMEOUT = 20
FALLBACK_MSG = "Hiện tại thư viện chưa có thông tin chính xác cho câu này. Bạn mô tả rõ hơn giúp mình nhé."

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

# --- COPY OF RERANK FUNCTION ---
def rerank_with_llm(user_q, candidates):
    if not candidates:
        return None
        
    # Build prompt exactly as in chat.py
    block = ""
    for i, c in enumerate(candidates, start=1):
        block += f"{i}. {c['answer']}\n"

    prompt = f"""
Bạn là chuyên gia tư vấn thông minh.
Nhiệm vụ: Tìm câu trả lời TỐT NHẤT cho câu hỏi của người dùng trong danh sách bên dưới.

Câu hỏi: "{user_q}"

Danh sách ứng viên:
{block}

HƯỚNG DẪN TƯ DUY (QUAN TRỌNG):
1. **XỬ LÝ CÂU HỎI LIỆT KÊ (LIST ALL) - QUAN TRỌNG**:
   - Nếu hỏi "Gồm những gì?", "Có những phòng nào?", "Liệt kê...", "Chia thành...".
   - -> BẮT BUỘC chọn câu trả lời có chứa DANH SÁCH (dấu gạch đầu dòng "-") hoặc từ "gồm", "bao gồm".
   - Ví dụ: Hỏi "Thư viện gồm những phòng nào?" -> Chọn câu "các phòng thư viện: - Phòng A... - Phòng B...".

2. **XỬ LÝ TÌM KIẾM CỤ THỂ (SPECIFIC LOOKUP)**:
   - Nếu hỏi trúng tên một phòng cụ thể (ví dụ: "Phòng mượn sách").
   - -> Hãy tìm trong danh sách xem có mục đó không. Nếu có -> CHỌN NGAY.

3. **SO KHỚP TỪ KHÓA & NGỮ NGHĨA**:
   - Hỏi "Ở đâu", "Chỗ nào" -> Tìm câu chứa địa điểm (Nhà, Phòng, Tầng, Lầu, Khu, Vị trí...).
   - Hỏi "Bao nhiêu", "Số lượng" -> Tìm câu chứa con số hoặc từ chỉ lượng (cuốn, bản, đầu sách...).
   - Hỏi "Thời gian", "Bao lâu" -> Tìm câu chứa ngày, giờ, tháng, năm.

4. **KIỂM TRA ĐỊNH DẠNG**:
   - Hỏi "Số điện thoại" -> Câu trả lời PHẢI có dãy số.
   - Hỏi "Link/Facebook" -> Câu trả lời PHẢI có "http".

KẾT QUẢ:
- Nếu tìm thấy câu trả lời phù hợp: Trả về SỐ THỨ TỰ (ví dụ: 1, 2...).
- Nếu không có câu nào khớp: Trả về 0.

Chỉ trả về 1 con số duy nhất.
"""
    out = llm(prompt, temp=0.1, n=10).strip()
    
    import re
    match = re.search(r'\d+', out)
    if match:
        idx = int(match.group()) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    return None

# --- COPY OF STRICT ANSWER FUNCTION ---
def strict_answer(question: str, knowledge: str) -> str:
    prompt = f"""
Bạn là trợ lý ảo của thư viện. 
NHIỆM VỤ: Trả lời câu hỏi dựa trên thông tin cung cấp bên dưới.

THÔNG TIN (KNOWLEDGE):
{knowledge}

CÂU HỎI (QUESTION): "{question}"

QUY TẮC BẮT BUỘC:
1. ⚠️ TUYỆT ĐỐI TRẢ LỜI BẰNG TIẾNG VIỆT. (Không dùng tiếng Trung/Anh).
2. Nếu thông tin có vẻ liên quan, HÃY TRẢ LỜI NGAY (đừng sợ sai).
3. Nếu thông tin là danh sách, hãy trích xuất ý chính.
4. Nếu câu hỏi dùng từ đồng nghĩa (ví dụ: "rách" = "hỏng"), hãy tự suy luận để trả lời.
5. Nếu có số liệu/thống kê, hãy đưa ra con số đó.
6. Nếu câu hỏi về đối tượng cụ thể (ví dụ: "sách tham khảo") mà thông tin chỉ nói chung chung (ví dụ: "sách"), HÃY DÙNG THÔNG TIN CHUNG ĐÓ để trả lời.
7. Nếu thông tin là SỐ ĐIỆN THOẠI, EMAIL, LINK -> Hãy trả lời ngay (ví dụ: "0987654321").
8. Nếu thông tin là QUY TRÌNH (Trình thẻ, Quét mã...) -> Hãy trả lời các bước đó.
9. Tuyệt đối KHÔNG trả lời "{FALLBACK_MSG}" nếu bạn tìm thấy thông tin liên quan dù chỉ một chút.

Nếu thông tin HOÀN TOÀN KHÔNG LIÊN QUAN thì mới nói: "{FALLBACK_MSG}"

Ví dụ:
- Info: "Mất sách đền gấp đôi" -> Hỏi: "Làm rách bị phạt ko?" -> Trả lời: "Có, bạn phải đền gấp đôi."
- Info: "0262.3825180" -> Hỏi: "Số nào?" -> Trả lời: "0262.3825180"
- Info: "Trình thẻ và tài liệu..." -> Hỏi: "Cách trả sách?" -> Trả lời: "Bạn cần trình thẻ và tài liệu cho cán bộ."

Câu trả lời của bạn (Tiếng Việt):
"""
    out = llm(prompt, temp=0.3, n=256) 
    if not out:
        return FALLBACK_MSG

    out = out.strip()
    if any(c.isdigit() for c in out) or "@" in out or "http" in out:
        return out
    if "không có thông tin" in out.lower() and len(out) < 15: 
         return FALLBACK_MSG
    return out

# --- REPLICATE DATA LOADING LOGIC ---
print("Đang tải dữ liệu từ faq.db...")
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# FAQ
cur.execute("SELECT question, answer, category FROM faq WHERE approved = 1 OR approved IS NULL")
faq_rows = cur.fetchall()
FAQ_TEXTS = []
def normalize(x): return " ".join(x.lower().strip().split())

for q, a, cat in faq_rows:
    content = f"{cat or ''}: {a or ''}" 
    FAQ_TEXTS.append(normalize(content))

print("Đang tạo embedding (lần đầu sẽ hơi lâu)...")
FAQ_EMB = embed_model.encode(FAQ_TEXTS, normalize_embeddings=True)

def search_faq_candidates(q_vec_or_text, top_k=15):
    if isinstance(q_vec_or_text, str):
        q_vec = embed_model.encode(normalize(q_vec_or_text), normalize_embeddings=True)
    else:
        q_vec = q_vec_or_text
        
    sims = np.dot(FAQ_EMB, q_vec)
    idx = np.argsort(-sims)[:top_k]
    candidates = []
    for i in idx:
        score = float(sims[i])
        if score < 0.10: continue
        q, a, cat = faq_rows[i]
        candidates.append({"score": score, "question": q, "answer": a, "category": cat})
    return candidates

# ------------------------------------

def debug_query(query):
    print(f"\n{'='*50}")
    print(f"QUERY: {query}")
    
    # 1. Search
    candidates = search_faq_candidates(query, top_k=15)
    print(f"\n[Search] Found {len(candidates)} candidates.")
    
    # Check if expected answers are in candidates
    # We expect "các phòng thư viện" for room queries
    # We expect "Trả sách" for procedure queries
    found_room_list = False
    found_procedure = False
    
    print("[Top 5 Candidates]:")
    for i, c in enumerate(candidates[:5]):
        print(f"  {i+1}. [{c['category']}] {c['answer'][:100]}...")
        if "các phòng thư viện" in c['answer'].lower():
            found_room_list = True
        if "trình thẻ" in c['answer'].lower():
            found_procedure = True
            
    if "phòng" in query.lower() and not found_room_list:
        print("⚠️ WARNING: 'List of rooms' answer NOT found in Top 5!")
        # Check in top 15
        for c in candidates:
            if "các phòng thư viện" in c['answer'].lower():
                print("  -> But found in Top 15.")
                break
        else:
             print("  -> NOT FOUND in Top 15 either. RETRIEVAL FAILURE.")

    # 2. Rerank
    selected = rerank_with_llm(query, candidates)
    if selected:
        print(f"\n[Rerank] ✅ Selected: [{selected['category']}] {selected['answer']}")
        
        # 3. Strict Answer
        final_ans = strict_answer(query, selected['answer'])
        print(f"\n[Strict Answer]: {final_ans}")
    else:
        print("\n[Rerank] ❌ No selection.")

queries = [
    "Phòng mượn sách ở đâu?",
    # "Hướng dẫn trả sách tại thư viện",
    # "Ngoài quản lý tài liệu thì thư viện còn những nhiệm vụ gì ?",
    # "Sách giáo trình có thể mượn theo môn học không ?",
    # "Phòng mấy tra cứu ở tầng bao nhiêu ?",
    # "Thư viện gồm những phòng nào ?"
]

for q in queries:
    debug_query(q)
