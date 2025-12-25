# ==========================================
# HO TÊN: Đỗ Thị Hồng Điệp
# MSSV: 23103014
# ĐỒ ÁN: Chatbot Dynamic Router - TTN University
# NGÀY NỘP: 21/12/2025
# Copyright © 2025. All rights reserved.
# ==========================================

# ============================================
#  CHATBOT 4-BƯỚC – HIỂU NGHĨA, KHÔNG BỊA
#  PHIÊN BẢN TỐI ƯU RAM
# ============================================

import os
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
# Load .env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "rag", ".env")

try:
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH, override=True)
    else:
        load_dotenv()
except Exception:
    pass

ZIPUR_API_KEY = os.getenv("ZIPUR_API_KEY")
ZIPUR_MODEL = os.getenv("ZIPUR_MODEL", "glm-4-plus")

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
#normalized_text = normalize(text)
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
    if not ZIPUR_API_KEY:
        return ""

    headers = {
        "Authorization": f"Bearer {ZIPUR_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": ZIPUR_MODEL,
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

# ============================================
def rerank_with_llm(user_q: str, candidates: list, context_str: str = ""):
    """✅ Giảm max_tokens từ 128 xuống 64"""
    if not candidates:
        return None

    # ✅ Chỉ rerank top 3 (giảm từ 5) để nhanh hơn
    top_candidates = candidates[:3]
    
    block = ""
    for i, c in enumerate(top_candidates, start=1):
        block += f"{i}. [{c['category']}] {c['answer']}\n"
    #tiêm trí nhớ
    
    context_section = ""
    if context_str:
        context_section = f"\nLịch sử hội thoại:\n{context_str}\n"
    prompt = f"""
Bạn là chuyên gia tư vấn thông minh.
Nhiệm vụ: Tìm câu trả lời PHÙ HỢP NHẤT cho câu hỏi của người dùng trong danh sách bên dưới.
{context_section}
Câu hỏi: "{user_q}"

Danh sách ứng viên:
{block}

HƯỚNG DẪN TƯ DUY:
- Hãy hiểu Ý NGHĨA của câu hỏi (không chỉ bắt từ khóa).
- Nếu có lịch sử hội thoại, sử dụng context để hiểu câu hỏi tốt hơn.
- Ví dụ: Hỏi "Fanpage" thì câu chứa "Facebook" là đúng. Hỏi "Quy trình" thì câu hướng dẫn các bước là đúng.
- Nếu câu hỏi tìm "Địa điểm" (ở đâu), hãy chọn câu chứa thông tin vị trí.
- Nếu câu hỏi tìm "Danh sách" (gồm những gì), hãy chọn câu liệt kê đầy đủ nhất.

YÊU CẦU:
- Nếu tìm thấy câu trả lời phù hợp: Trả về SỐ THỨ TỰ (ví dụ: 1, 2...).
- Nếu không có câu nào khớp: Trả về 0.

Chỉ trả về 1 con số duy nhất.
"""
    out = llm(prompt, temp=0.1, n=32).strip()  # ← Giảm từ 64 xuống 32

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



#  MAIN PROCESS - DYNAMIC & AUTOMATED
# ============================================
def process_message(text: str, history: list = None, image_path: str = None) -> str:
    """
    DYNAMIC VERSION + Multi-step Reasoning + Conversation Memory
    - Router ngữ nghĩa (Vector + LLM CoT)
    - Clarification (hỏi lại khi mơ hồ)
    - Search theo collection
    - Humanize answer (chỉ học từ CÂU TRẢ LỜI)
    - Conversation memory (nhớ 2-3 câu trước)
    """
    print("[CHAT.PY] ĐÃ GỌI NÃO (Dynamic Reasoning Mode)")

    if not text.strip():
        return "Xin chào 👋 Bạn muốn hỏi thông tin gì trong thư viện?"
    
    # Bước 1  lấy lịch sử gần nhất
    context_str = ""
    if history:
        context_lines = []
        for user_msg, bot_msg in history:
            context_lines.append(f"User: {user_msg}")
            context_lines.append(f"Bot: {bot_msg}")
        context_str = "\n".join(context_lines)
        print(f"[CONTEXT] Using {len(history)} previous messages")
# lấy cặp đóng gói1 phía trên 196 phía dưới 296
    try:
        # Import dynamic tools (đã sửa ở trên)
        from chat_dynamic_router import (
            reason_and_route,
            search_dynamic,
            get_collections_with_descriptions,
            humanize_answer,
            

        )

        # ✅ Lấy model (lazy load)
        model = get_model()

        # chuẩn hoá và tạo vector
        normalized_text = normalize(text)
        q_vec = model.encode(normalized_text, normalize_embeddings=True)
        #reason_and_route
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
       #Nơi chuẩn bị dữ liệu
        router_question = text
        if context_str:
            router_question = f"{text}\n\n[Lịch sử gần đây:\n{context_str}]"
        router_result = reason_and_route(router_question, q_vec, llm, model)

        # Nếu cần hỏi lại → trả luôn câu hỏi clarify (không search)
        if router_result.needs_clarification and router_result.clarification_question:
            print("[PROCESS] Clarification required → hỏi lại người dùng.")
            return router_result.clarification_question


        # BƯỚC 6 – Search đúng collection (có lọc ngành nếu cần)
        rewritten = router_result.rewritten_question or text

    
        q_vec_search = model.encode(
            normalize(rewritten), normalize_embeddings=True
        )
      

        # B5: Search vào knowledge_base, filter theo collection nếu có
        collection_name = router_result.target_collection or "global"
        print(f"[PROCESS] Search in collection: {collection_name}")
        
        # ✅ B5a: Sử dụng Dynamic Filter từ Router (nếu có)
    
        candidates = search_dynamic(
            collection_name, 
            q_vec_search, 
            top_k=10, 
            
        )

        if not candidates:
            print("[DEBUG] ❌ Không tìm thấy kết quả nào.")
            return "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp trong cơ sở dữ liệu."

        print(f"[DEBUG] Found {len(candidates)} candidates.")
        for c in candidates:
            print(
                f"  - [{c['score']:.4f}] {c['answer'][:80]}... (Cat: {c['category']})"
            )

        # B6a: Dùng LLM để hiểu user muốn bao nhiêu kết quả
        extract_prompt = f"""
Phân tích câu hỏi sau và trả lời:

Câu hỏi: "{text}"

Hỏi:
1. User có thực sự muốn hỏi danh sách NHIỀU kết quả không (ví dụ: "liệt kê", "các loại", "những cuốn", "top 5")? (có/không)
2. Nếu có, user muốn bao nhiêu kết quả? (trả số, nếu không rõ thì trả 1)

Chỉ trả lời theo format: <có/không>|<số>

Ví dụ:
- "Gợi ý các sách về Python" → có|3
- "Cho tôi 5 cuốn về AI" → có|5
- "Sách Python giá bao nhiêu?" → không|1
- "Sách python" → không|1
- "Thông tin về sách Java" → không|1
"""
        
        try:
            llm_response = llm(extract_prompt, temp=0.1, n=20).strip()
            parts = llm_response.split('|')
            
            if len(parts) == 2 and parts[0].lower() == 'có':
                try:
                    requested_count = int(parts[1])
                    print(f"[DEBUG] 🔢 LLM phát hiện: User muốn {requested_count} kết quả")
                    
                    # Lấy đúng số lượng user yêu cầu (tối đa 10)
                    actual_count = min(requested_count, len(candidates), 10)
                    top_n = candidates[:actual_count * 2]  # Lấy gấp đôi để lọc
                    
                    # ✅ LLM Filter: Lọc chỉ giữ kết quả liên quan
                    filter_prompt = f"""
Câu hỏi: "{text}"
Danh sách kết quả:
{chr(10).join([f"{i+1}. {c['answer'][:200]}" for i, c in enumerate(top_n)])}

NHIỆM VỤ: Chọn {requested_count} kết quả THỰC SỰ LIÊN QUAN đến câu hỏi.

QUY TẮC NGHIÊM NGẶT:
- Nếu hỏi về "công nghệ thông tin" → CHỈ chọn sách về lập trình, AI, dữ liệu, máy tính
- LOẠI BỎ sách về: ngôn ngữ, toán học cơ bản, vật lý, hóa học (trừ khi câu hỏi yêu cầu)
- Ưu tiên sách có từ khóa CHÍNH XÁC khớp với câu hỏi

Trả về danh sách số thứ tự (ví dụ: 2,5,7), KHÔNG giải thích:
"""
                    try:
                        filter_response = llm(filter_prompt, temp=0.1, n=30).strip()
                        selected_indices = [int(x.strip())-1 for x in filter_response.split(',') if x.strip().isdigit()]
                        selected_candidates = [top_n[i] for i in selected_indices if 0 <= i < len(top_n)]
                        
                        if selected_candidates:
                            top_n = selected_candidates[:requested_count]
                            print(f"[DEBUG] 🔍 LLM filtered: Giữ {len(top_n)} kết quả liên quan")
                        else:
                            top_n = top_n[:requested_count]  # Fallback
                    except:
                        top_n = top_n[:requested_count]  # Fallback nếu filter lỗi
                    
                    combined_answer = "\n\n".join([
                        f"{i+1}. {c['answer']}" 
                        for i, c in enumerate(top_n)
                    ])
                    print(f"[DEBUG] ✅ Trả về {actual_count} kết quả")
                    print(f"[DEBUG] 📝 Raw answer (before humanize):")
                    print(combined_answer)
                    print("[DEBUG] ==================")
                    final_ans = humanize_answer(text, combined_answer)
                    print(f"[DEBUG] 🎨 After humanize:")
                    print(final_ans)
                    print("[DEBUG] ==================")
                    return final_ans
                except ValueError:
                    pass
        except Exception as e:
            print(f"[DEBUG] ⚠️ LLM extract failed: {e}, falling back to single-result rerank")

        # B6b: Rerank với LLM (Chọn câu trả lời phù hợp nhất) - Chỉ khi hỏi 1 câu cụ thể
        best_cand = rerank_with_llm(rewritten, candidates, context_str=context_str)

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

        # B7: HUMANIZE ANSWER (viết lại tự nhiên)
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
if __name__ == "__main__":
    print("🤖 Chatbot 4-BƯỚC (Phiên bản TỐI ƯU RAM) đã sẵn sàng!")
    while True:
        q = input("\nBạn: ")
        if q.lower() in ["quit", "bye", "exit", "thoát"]:
            print("Hẹn gặp lại bạn ở thư viện nhé! 📚")
            break
        print("Bot:", process_message(q))

# User question
#  → embed
#  → router chọn collection
#  
#  → rewrite câu hỏi
#  → embed lại
#  → search_dynamic  ← GIỮ DÒNG NÀY
#  → rerank_with_llm
#  → humanize_answer

