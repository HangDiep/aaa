# ============================================
#  CHATBOT 4-BƯỚC – HIỂU NGHĨA, KHÔNG BỊA
#  PHIÊN BẢN TỐI ƯU RAM
# ============================================

import os
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import json
import re
import time
import random
import gc  # ✅ Garbage collector
from dotenv import load_dotenv

# Load .env
ENV_PATH = r"D:\HTML\a_Copy\rag\.env"
try:
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH, override=True)
    else:
        load_dotenv()
except Exception:
    pass

FAQ_DB_PATH = os.getenv("FAQ_DB_PATH")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "glm-4-plus")

FALLBACK_MSG = "Hiện tại thư viện chưa có thông tin chính xác cho câu này. Bạn mô tả rõ hơn giúp mình nhé."

# ============================================
#  EMBEDDING MODEL - LAZY LOAD + AUTO CLEANUP
# ============================================
embed_model = None
last_model_use = 0
MODEL_TIMEOUT = 300  # ✅ Giải phóng model sau 5 phút không dùng

def get_model():
    global embed_model, last_model_use
    
    if embed_model is not None:
        last_model_use = time.time()
        return embed_model
    
    try:
        print("🔄 Đang load model BAAI/bge-m3...")
        embed_model = SentenceTransformer("BAAI/bge-m3")
        print("✅ Load BAAI/bge-m3 thành công!")
    except Exception as e:
        print(f"⚠ Lỗi load BAAI/bge-m3: {e}")
        print("🔄 Đang dùng fallback model keepitreal/vietnamese-sbert...")
        embed_model = SentenceTransformer("keepitreal/vietnamese-sbert")
        print("✅ Load fallback thành công!")
    
    last_model_use = time.time()
    return embed_model

def cleanup_model_if_idle():
    """✅ Giải phóng model nếu không dùng lâu"""
    global embed_model, last_model_use
    if embed_model is not None and (time.time() - last_model_use) > MODEL_TIMEOUT:
        print("🧹 Giải phóng embedding model (idle quá lâu)...")
        del embed_model
        embed_model = None
        gc.collect()

# ============================================
#  TEXT NORMALIZE
# ============================================
def normalize(x: str) -> str:
    return " ".join(x.lower().strip().split())

# ============================================
#  LLM CALL - TỐI ƯU HÓA
# ============================================
def llm(prompt: str, temp: float = 0.15, n: int = 1024) -> str:
    """
    Gọi Zhipu AI API với retry logic
    ✅ Giảm timeout, giảm max_tokens mặc định
    """
    if not GROQ_API_KEY:
        return ""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "max_tokens": n,
    }

    max_retries = 2  # ✅ Giảm từ 3 xuống 2
    base_delay = 1   # ✅ Giảm từ 2s xuống 1s

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                headers=headers,
                json=payload,
                timeout=20  # ✅ Giảm từ 30s xuống 20s
            )
            
            if resp.status_code == 200:
                data = resp.json()
                result = data["choices"][0]["message"]["content"].strip()
                del data  # ✅ Giải phóng response data
                return result
            
            if resp.status_code == 429:
                wait_time = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                print(f"⚠ Zhipu AI quá tải (429). Đang chờ {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
                
            print(f"⚠ Lỗi Zhipu AI {resp.status_code}: {resp.text}")
            return ""

        except Exception as e:
            print(f"⚠ Lỗi gọi Zhipu AI: {e}")
            return ""
    
    return ""

# ============================================
#  CONNECT TO QDRANT - LAZY INIT
# ============================================
from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

qdrant_client = None

def get_qdrant_client():
    """✅ Lazy initialization cho Qdrant client"""
    global qdrant_client
    if qdrant_client is None:
        print("🔗 Kết nối tới Qdrant...")
        if QDRANT_API_KEY:
            qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            qdrant_client = QdrantClient(url=QDRANT_URL)
        
        try:
            collections = qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            print(f"✅ Đã kết nối Qdrant: {len(collections)} collections ({', '.join(collection_names)})")
        except Exception as e:
            print(f"❌ Lỗi kết nối Qdrant: {e}")
    
    return qdrant_client

# ============================================
#  ROUTER - TỐI ƯU HÓA
# ============================================

def is_greeting(text: str) -> bool:
    t = text.lower().strip()
    greet_words = ["xin chào", "chào bạn", "chào ad", "hello", "hi", "alo"]
    return any(w in t for w in greet_words)



# ============================================
#  REWRITE - TỐI ƯU HÓA
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
- "số nao" -> "số điện thoại liên hệ hotline là gì"
- "mở cửa ko" -> "giờ mở cửa hoạt động như thế nào"
- "liên hệ sao" -> "cách thức liên hệ với thư viện"

Câu gốc: "{q}"

Câu viết lại (chỉ viết 1 câu duy nhất):
"""
    out = llm(prompt, temp=0.1, n=64)
    return out.strip() if out else q
# ============================================
#  RERANK - TỐI ƯU HÓA
# ============================================
def rerank_with_llm(user_q: str, candidates: list):
    """✅ Giảm max_tokens từ 128 xuống 64"""
    if not candidates:
        return None

    # ✅ Chỉ rerank top 5 thay vì tất cả
    top_candidates = candidates[:5]
    
    block = ""
    for i, c in enumerate(top_candidates, start=1):
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
    out = llm(prompt, temp=0.1, n=64).strip()

    match = re.search(r'\d+', out)
    if match:
        idx = int(match.group()) - 1
        if 0 <= idx < len(top_candidates):
            return top_candidates[idx]

    # Fallback: tin top 1 nếu score rất cao
    if top_candidates and top_candidates[0]['score'] > 0.45:
        print(f"[Rerank] LLM từ chối, nhưng Top 1 score cao ({top_candidates[0]['score']:.2f}) -> Chọn Top 1.")
        return top_candidates[0]

    return None

# ============================================
#  STRICT ANSWER - TỐI ƯU HÓA
# ============================================
def strict_answer(question: str, knowledge: str) -> str:
    """✅ Giảm max_tokens từ 128 xuống 120 (cân bằng RAM vs chất lượng)"""
    print(f"[DEBUG STRICT] Q: {question} | Knowledge: {knowledge[:50]}...")
    prompt = f"""
Bạn là trợ lý ảo của thư viện. Trả lời NGẮN GỌN, ĐÚNG TRỌNG TÂM.

THÔNG TIN:
{knowledge}

CÂU HỎI: "{question}"

QUY TẮC:
1. Trả lời NGẮN (1-2 câu), chỉ thông tin CHÍNH XÁC từ KNOWLEDGE
2. KHÔNG thêm lời chào, KHÔNG hỏi lại, KHÔNG giải thích dài dòng
3. Nếu hỏi về email/hotline/facebook → CHỈ trả thông tin đó, KHÔNG thêm gì khác
4. Nếu KNOWLEDGE không liên quan → Trả: "{FALLBACK_MSG}"

VÍ DỤ:
Q: "email thư viện"
K: "Email: thuvien@ttn.edu.vn, Hotline: 0123456789"
A: "Email của thư viện là thuvien@ttn.edu.vn nhé!"

Q: "facebook thư viện"
K: "Email: thuvien@ttn.edu.vn, Hotline: 0123456789"
A: "{FALLBACK_MSG}"

Trả lời (NGẮN GỌN):
"""
    out = llm(prompt, temp=0.1, n=120)
    print(f"[DEBUG STRICT OUT] {out}")

    if not out:
        return FALLBACK_MSG

    out = out.strip()
    
    # Loại bỏ câu hỏi thừa ở cuối
    if "?" in out:
        sentences = out.split("?")
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            out = sentences[0].strip() + "."
    
    # Loại bỏ lời chào thừa
    greetings = ["Chào bạn!", "Xin chào!", "Dạ,", "Vâng,"]
    for g in greetings:
        if out.startswith(g):
            out = out[len(g):].strip()
    
    # Chấp nhận câu trả lời có số / email / link
    if any(c.isdigit() for c in out) or "@" in out or "http" in out:
        return out

    if "không có thông tin" in out.lower() and len(out) < 15:
        return FALLBACK_MSG

    return out

# ============================================
#  MAIN PROCESS - TỐI ƯU HÓA
# ============================================
# ============================================
#  MAIN PROCESS - DYNAMIC & AUTOMATED
# ============================================
def process_message(text: str) -> str:
    """
    DYNAMIC VERSION + Multi-step Reasoning
    - Router ngữ nghĩa (Vector + LLM CoT)
    - Clarification (hỏi lại khi mơ hồ)
    - Search theo collection
    - Humanize answer (chỉ học từ CÂU TRẢ LỜI)
    """
    print("[CHAT.PY] ĐÃ GỌI NÃO (Dynamic Reasoning Mode)")

    if not text.strip():
        return "Xin chào 👋 Bạn muốn hỏi thông tin gì trong thư viện?"

    try:
        # Import dynamic tools (đã sửa ở trên)
        from chat_dynamic_router import (
            reason_and_route,
            search_dynamic,
            get_collections_with_descriptions,
            humanize_answer,
            GLOBAL_COLLECTION,
        )

        # ✅ Lấy model (lazy load)
        model = get_model()

        # B0: Tạo vector 1 lần duy nhất
        normalized_text = normalize(text)
        q_vec = model.encode(normalized_text, normalize_embeddings=True)

        # B1: Greeting
        if is_greeting(text) and len(text.split()) <= 4:
            collections = get_collections_with_descriptions()
            collection_names = ", ".join(
                [n.upper() for n in list(collections.keys())[:3]]
            )
            return (
                f"Xin chào! Tôi là trợ lý ảo. Bạn có thể hỏi về: "
                f"{collection_names} hoặc bất cứ thông tin nào khác..."
            )

        # B2: Multi-step Reasoning Router (CoT + Clarification)
        router_result = reason_and_route(text, q_vec, llm, model)

        # Nếu cần hỏi lại → trả luôn câu hỏi clarify (không search)
        if router_result.needs_clarification and router_result.clarification_question:
            print("[PROCESS] Clarification required → hỏi lại người dùng.")
            return router_result.clarification_question

        # B3: Lấy câu hỏi đã làm rõ (rewritten_question)
        rewritten = router_result.rewritten_question or text

        # Tùy chọn: nếu bạn vẫn muốn thêm lớp rewrite_question cũ
        # rewritten2 = rewrite_question(rewritten)
        # if rewritten2: rewritten = rewritten2

        # B4: Embed lại cho search
        q_vec_search = model.encode(
            normalize(rewritten), normalize_embeddings=True
        )

        # B5: Search vào knowledge_base, filter theo collection nếu có
        collection_name = router_result.target_collection or "global"
        print(f"[PROCESS] Search in collection: {collection_name}")
        candidates = search_dynamic(collection_name, q_vec_search, top_k=10)

        if not candidates:
            print("[DEBUG] ❌ Không tìm thấy kết quả nào.")
            return "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp trong cơ sở dữ liệu."

        print(f"[DEBUG] Found {len(candidates)} candidates.")
        for c in candidates:
            print(
                f"  - [{c['score']:.4f}] {c['answer'][:80]}... (Cat: {c['category']})"
            )

        # B6: Rerank với LLM (Chọn câu trả lời phù hợp nhất)
        best_cand = rerank_with_llm(rewritten, candidates)

        if not best_cand:
            if candidates and candidates[0]["score"] > 0.35:
                best_cand = candidates[0]
                print(
                    "[DEBUG] ⚠️ Rerank từ chối, nhưng lấy Top 1 do score ổn."
                )
            else:
                print("[DEBUG] ❌ Rerank từ chối tất cả.")
                return (
                    "Xin lỗi, tôi tìm thấy một số thông tin nhưng có vẻ không khớp với câu hỏi của bạn."
                )
        else:
            print(
                f"[DEBUG] ✅ Rerank chọn: {best_cand['answer'][:80]}..."
            )

        # B7: HUMANIZE ANSWER (chỉ học từ CÂU TRẢ LỜI)
        raw_answer = best_cand["answer"]
        final_ans = humanize_answer(text, raw_answer)
        return final_ans

    except Exception as e:
        print(f"[PROCESS] ❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return "Xin lỗi, hệ thống đang gặp lỗi xử lý. Vui lòng thử lại sau."

    finally:
        gc.collect()
        cleanup_model_if_idle()


# ============================================
#  CLI
# ============================================
if __name__ == "__main__":
    print("🤖 Chatbot 4-BƯỚC (Phiên bản TỐI ƯU RAM) đã sẵn sàng!")
    while True:
        q = input("\nBạn: ")
        if q.lower() in ["quit", "bye", "exit", "thoát"]:
            print("Hẹn gặp lại bạn ở thư viện nhé! 📚")
            break
        print("Bot:", process_message(q))
