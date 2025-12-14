"""
Enterprise Dynamic Router & Search
Multi-step Reasoning Router + Clarification + Humanize Answer
Query vào Single Collection 'knowledge_base' với Metadata Filters
"""

import sqlite3
import time
import numpy as np
from typing import Dict, List, Optional
import os
from dataclasses import dataclass
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

FAQ_DB_PATH = os.getenv("FAQ_DB_PATH", os.path.join(BASE_DIR, "faq.db"))
GLOBAL_COLLECTION = "knowledge_base"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "glm-4-plus")



# ============================================
#  COLLECTIONS CONFIG CACHE
# ============================================

# Global cache for dynamic column mappings
_column_mappings_cache: Dict[str, Dict[str, str]] = {}

def get_collections_with_descriptions() -> Dict[str, str]:
    """
    Đọc danh sách bảng & mô tả từ SQLite (collections_config).
    Đồng thời load column_mappings vào cache.
    """
    import json
    
    global _collections_cache, _cache_time, _column_mappings_cache

    # cache 5 phút
    if _collections_cache and time.time() - _cache_time < CACHE_TTL:
        return _collections_cache

    try:
        conn = sqlite3.connect(FAQ_DB_PATH)
        cur = conn.cursor()
        
        # Ensure table exists (sync_dynamic might not have run yet)
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='collections_config'")
        if not cur.fetchone():
            return {}

        cur.execute("SELECT name, description, column_mappings FROM collections_config WHERE enabled=1")
        rows = cur.fetchall()
        conn.close()

        collections = {}
        for row in rows:
            tbl_name = row[0]
            desc = row[1]
            raw_map = row[2]
            
            collections[tbl_name] = desc
            
            # Load mapping if available
            if raw_map:
                try:
                    _column_mappings_cache[tbl_name] = json.loads(raw_map)
                except:
                    pass

        _collections_cache = collections
        _cache_time = time.time()

        print(f"[ROUTER] Loaded {len(collections)} collections from SQLite.")
        return collections

    except Exception as e:
        print("[ROUTER] ERROR reading collections_config:", e)
        return {}


def get_description_embeddings(model) -> Dict[str, np.ndarray]:
    """
    Cache embedding của các mô tả collection để so sánh nhanh.
    Mỗi collection có 1 vector đại diện: "name: description"
    """
    global _description_embeddings_cache

    collections = get_collections_with_descriptions()
    if not collections:
        return {}

    if (
        _description_embeddings_cache
        and len(_description_embeddings_cache) == len(collections)
    ):
        return _description_embeddings_cache

    print("[ROUTER] Caching collection description embeddings...")
    texts: List[str] = []
    names: List[str] = []

    for name, desc in collections.items():
        text = f"{name}: {desc}"
        texts.append(text)
        names.append(name)

    try:
        embeddings = model.encode(texts, normalize_embeddings=True)
    except Exception as e:
        print(f"[ROUTER] Error encoding collection descriptions: {e}")
        return {}

    _description_embeddings_cache = {}
    for i, name in enumerate(names):
        _description_embeddings_cache[name] = np.array(embeddings[i])

    return _description_embeddings_cache


# ============================================
#  LLM NỘI BỘ (dùng cho humanize nếu cần)
# ============================================

def _local_llm(prompt: str, temp: float = 0.2, n: int = 256) -> str:
    if not GROQ_API_KEY:
        return ""

    import requests
    import random

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "max_tokens": n,
    }

    max_retries = 2
    base_delay = 1

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                headers=headers,
                json=payload,
                timeout=20,
            )
            if resp.status_code == 200:
                data = resp.json()
                result = data["choices"][0]["message"]["content"].strip()
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


def humanize_answer(user_question: str, raw_answer: str) -> str:
    """
    Viết lại câu trả lời cho tự nhiên, thân thiện như nhân viên thư viện.
    CHỈ HỌC TỪ CÂU TRẢ LỜI (raw_answer). Câu hỏi chỉ để tham chiếu ngữ cảnh.
    """
    # Đếm số items trong raw_answer
    item_count = raw_answer.count('\n\n') + 1 if '\n\n' in raw_answer else 1
    
    prompt = f"""
Bạn là nhân viên thư viện, nhiệm vụ là viết lại câu trả lời ngắn gọn nhưng tự nhiên, thân thiện.

THÔNG TIN CHÍNH XÁC (CHỈ ĐƯỢC DÙNG DỮ LIỆU NÀY):
{raw_answer}

CÂU HỎI: "{user_question}"

YÊU CẦU NGHIÊM NGẶT:
1. Raw answer có {item_count} items → BẠN PHẢI LIỆT KÊ ĐẦY ĐỦ {item_count} items
2. NGHIÊM CẤM bỏ qua bất kỳ item nào
3. NGHIÊM CẤM viết "chỉ tìm thấy 1" khi có {item_count} items
4. Nếu có format "1. ... 2. ... 3. ..." → Giữ nguyên cấu trúc danh sách
5. Chuyển "key: value" thành câu văn tự nhiên
6. Giữ thái độ lịch sự

VÍ DỤ (3 items):
Raw: "1. name: Python\n\n2. name: Java\n\n3. name: C++"
→ "Thư viện có 3 sách:
1. Python
2. Java
3. C++"

PHẢI ĐỦ {item_count} ITEMS! KHÔNG ĐƯỢC BỎ QUA!

Trả lời:
"""
    out = _local_llm(prompt, temp=0.1, n=400)  # Tăng tokens + giảm temp
    return out.strip() if out else raw_answer.strip()


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# ============================================
#  ROUTER RESULT STRUCT
# ============================================

@dataclass
class RouterResult:
    target_collection: Optional[str]       # tên bảng hoặc None (GLOBAL)
    mode: str                              # "GLOBAL" hoặc "COLLECTION"
    rewritten_question: str                # câu hỏi đã làm rõ
    needs_clarification: bool              # có cần hỏi lại user không
    clarification_question: Optional[str]  # câu hỏi để hỏi lại
    confidence: float                      # độ tự tin (0-1)


# ============================================
#  MULTI-STEP REASONING ROUTER (CoT + Clarification)
# ============================================

def reason_and_route(
    question: str, q_vec: np.ndarray, llm_func, model
) -> RouterResult:
    """
    Multi-step Intent Reasoning + Clarification.

    1. Vector routing với embedding mô tả từng collection.
    2. Nếu đủ tự tin → chọn collection luôn.
    3. Nếu mơ hồ → LLM CoT:
        - Hiểu ý định
        - Chọn collection hoặc GLOBAL
        - Quyết định có cần hỏi lại không
        - Viết lại câu hỏi rõ nghĩa hơn
    """
    collections = get_collections_with_descriptions()
    if not collections:
        return RouterResult(
            target_collection=None,
            mode="GLOBAL",
            rewritten_question=question,
            needs_clarification=False,
            clarification_question=None,
            confidence=0.0,
        )

    # ---------- B1: VECTOR ROUTING ----------
    desc_embeds = get_description_embeddings(model)
    scores = []
    for name, emb in desc_embeds.items():
        s = cosine_similarity(q_vec, emb)
        scores.append((name, s))

    scores.sort(key=lambda x: x[1], reverse=True)
    best_name, best_score = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else 0.0
    margin = best_score - second_score

    print(
        f"[ROUTER] Vector → best={best_name} ({best_score:.3f}), "
        f"second={second_score:.3f}, margin={margin:.3f}"
    )

    # Ngưỡng tự tin: tin tưởng vector hơn để nhanh hơn
    if best_score > 0.70 and margin > 0.15:
        print(f"[ROUTER] ✅ Tin tưởng VECTOR, chọn collection: {best_name}")
        return RouterResult(
            target_collection=best_name,
            mode="COLLECTION",
            rewritten_question=question,
            needs_clarification=False,
            clarification_question=None,
            confidence=float(best_score),
        )

    # ---------- B2: LLM REASONING (CoT + Clarification) ----------
    top_candidates = scores[:3]
    options_lines = []
    for name, s in top_candidates:
        desc = collections.get(name, "")
        options_lines.append(f"- {name}: {desc} (similarity={s:.2f})")
    options_str = "\n".join(options_lines)

    prompt = f"""
Bạn là ROUTER thông minh cho chatbot thư viện.

CÂU HỎI GỐC:
\"{question}\"

CÁC BẢNG DỮ LIỆU (COLLECTIONS) CÓ THỂ LIÊN QUAN:
{options_str}

HƯỚNG DẪN QUAN TRỌNG:
- Nếu user hỏi về SÁCH (gợi ý sách, tìm sách, sách nào, có sách...) → LUÔN chọn "sch_"
- Nếu user hỏi về NGÀNH HỌC (ngành gì, mã ngành...) → Chọn "ngnh"
- Nếu user hỏi về SỐ LƯỢNG (bao nhiêu, thống kê, có bao nhiêu...) hoặc QUY MÔ → Chọn "faq_"
- Nếu user hỏi về FAQ (câu hỏi thường gặp, hỏi đáp...) → Chọn "faq_"

VÍ DỤ:
- "Gợi ý 3 sách về CNTT" → target_collection: "sch_"
- "Có sách Python không?" → target_collection: "sch_"
- "Tìm sách về AI" → target_collection: "sch_"
- "Ngành CNTT mã là gì?" → target_collection: "ngnh"
- "Thư viện mở cửa mấy giờ?" → target_collection: "faq_"

NHIỆM VỤ:
1. Hiểu người dùng đang hỏi về loại thông tin gì
2. Quyết định câu hỏi nên tra trong bảng nào
3. Nếu câu hỏi QUÁ MƠ HỒ → đề xuất hỏi lại
4. Viết lại câu hỏi rõ nghĩa hơn

ĐỊNH DẠNG TRẢ LỜI (JSON, KHÔNG GIẢI THÍCH THÊM):
{{
  "target_collection": "<tên collection hoặc null nếu GLOBAL>",
  "needs_clarification": true/false,
  "clarification_question": "<câu hỏi cần hỏi lại nếu needs_clarification=true>",
  "rewritten_question": "<phiên bản câu hỏi rõ nghĩa hơn>",
  "confidence": 0.0-1.0
}}

"""

    try:
        raw = llm_func(prompt, temp=0.2, n=256)
        import json
        clean = raw.strip()
        clean = clean.replace("```json", "").replace("```", "").strip()

        data = json.loads(clean)
        target_collection = data.get("target_collection")
        if isinstance(target_collection, str):
            target_collection = target_collection.strip() or None

        needs_clarification = bool(data.get("needs_clarification", False))
        clarification_question = data.get("clarification_question") or None
        rewritten_question = data.get("rewritten_question") or question
        confidence = float(data.get("confidence", 0.0))

        # Chuẩn hoá tên collection nếu LLM trả về dạng khác
        if target_collection:
            target_collection = target_collection.lower()
            valid = list(collections.keys())
            if target_collection not in valid:
                # Thử match kiểu case-insensitive
                for name in valid:
                    if target_collection == name.lower():
                        target_collection = name
                        break
                else:
                    # Không match được → dùng GLOBAL
                    target_collection = None

        mode = "COLLECTION" if target_collection else "GLOBAL"

        print(
            f"[ROUTER LLM] collection={target_collection}, "
            f"clarify={needs_clarification}, conf={confidence:.2f}"
        )

        return RouterResult(
            target_collection=target_collection,
            mode=mode,
            rewritten_question=rewritten_question,
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
            confidence=confidence,
        )

    except Exception as e:
        print(f"[ROUTER] ⚠️ LLM Reasoning error: {e}. Fallback GLOBAL.")
        return RouterResult(
            target_collection=None,
            mode="GLOBAL",
            rewritten_question=question,
            needs_clarification=False,
            clarification_question=None,
            confidence=float(best_score),
        )


# ============================================
#  BACKWARD-COMPAT: route_llm_dynamic (trả Filter)
# ============================================

def route_llm_dynamic(
    question: str, q_vec: np.ndarray, llm_func, model_func=None
) -> Optional[Dict]:
    """
    Giữ lại hàm cũ cho tương thích, nhưng bên trong dùng reason_and_route.
    Trả về:
      - Filter(source_table=...) nếu chọn 1 collection
      - None nếu GLOBAL
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    if model_func is None:
        # Không có model → coi như GLOBAL
        return None

    router_result = reason_and_route(question, q_vec, llm_func, model_func)

    if router_result.mode == "COLLECTION" and router_result.target_collection:
        return Filter(
            must=[
                FieldCondition(
                    key="source_table",
                    match=MatchValue(value=router_result.target_collection),
                )
            ]
        )
    else:
        return None


# ============================================
#  SEARCH DYNAMIC
# ============================================


def search_dynamic(
    collection_name: str, q_vec: np.ndarray, top_k: int = 10, ngành_id: int = None
) -> List[Dict]:
    """
    Search vào Qdrant bằng HTTP API trực tiếp (không dùng client.search)
    - Global collection: GLOBAL_COLLECTION
    - Nếu collection_name != 'faq' và != 'global' -> filter theo source_table
    - Nếu ngành_id được cung cấp -> filter thêm theo id_ngành
    """
    import requests
    import json

    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    url = f"{QDRANT_URL}/collections/{GLOBAL_COLLECTION}/points/search"

    # Body query gửi lên Qdrant
    body: Dict = {
        "vector": q_vec.tolist(),
        "limit": top_k,
        "with_payload": True,
        "with_vector": False,
        "score_threshold": 0.35,
    }

    # Filter theo collection_name nếu không phải global
    if collection_name and collection_name not in ("faq", "global"):
        must_conditions = [
            {
                "key": "source_table",
                "match": {"value": collection_name},
            }
        ]
        
        # Note: Không filter id_ngnh ở Qdrant vì cấu trúc nested array phức tạp
        # Sẽ filter trong Python sau khi nhận kết quả
        
        body["filter"] = {"must": must_conditions}

    headers = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("result", [])

        candidates: List[Dict] = []

        for hit in hits:
            score = hit.get("score", 0.0)
            payload = hit.get("payload") or {}

            source = (payload.get("source_table") or "general").upper()

            technical_fields = [
                "vector",
                "notion_id",
                "last_updated",
                "approved",
                "source_table",
            ]

            data_items = []
            
            # Get table-specific mapping
            table_map = _column_mappings_cache.get(source.lower(), {})
            
            for k, v in payload.items():
                if k not in technical_fields and v not in (None, ""):
                    # Dynamic mapping lookup with fallback to key itself
                    display_key = table_map.get(k, k)
                    data_items.append(f"{display_key}: {v}")

            final_content = " | ".join(data_items)

            question_context = (
                payload.get("question")
                or payload.get("title")
                or payload.get("name")
                or "Thông tin chi tiết"
            )
            
            # ✅ Extract id_ngnh từ nested array: [{"type": "number", "number": 1}]
            # ✅ Extract id_ngnh từ nested array: {"type": "array", "array": [{"type": "number", "number": 1}]}
            extracted_ngành_id = None
            id_ngnh_raw = payload.get("id_ngnh")
            
            # Debug extraction
            # print(f"[DEBUG-EXTRACT] id_ngnh raw type: {type(id_ngnh_raw)}")

            if id_ngnh_raw:
                try:
                    # Nếu là string JSON, parse ra dict/list
                    if isinstance(id_ngnh_raw, str):
                        import json
                        id_ngnh_raw = json.loads(id_ngnh_raw)
                    
                    # Case 1: Direct list (ít gặp trong structure này nhưng cứ giữ)
                    if isinstance(id_ngnh_raw, list) and len(id_ngnh_raw) > 0:
                        first_item = id_ngnh_raw[0]
                        if isinstance(first_item, dict):
                            extracted_ngành_id = first_item.get("number")
                            
                    # Case 2: Dict notion format {"type": "array", "array": [...]}
                    elif isinstance(id_ngnh_raw, dict):
                        if id_ngnh_raw.get("type") == "array" and "array" in id_ngnh_raw:
                            array_val = id_ngnh_raw.get("array")
                            if isinstance(array_val, list) and len(array_val) > 0:
                                first_item = array_val[0]
                                if isinstance(first_item, dict) and first_item.get("type") == "number":
                                    extracted_ngành_id = first_item.get("number")
                                    
                    # print(f"[DEBUG-EXTRACT] Extracted ID: {extracted_ngành_id}")
                except Exception as e:
                    print(f"[DEBUG-EXTRACT] Error: {e}")
                    pass

            candidates.append(
                {
                    "score": score,
                    "question": f"[{source}] {question_context}",
                    "answer": final_content,
                    "category": source,
                    "id": hit.get("id"),
                    "ngành_id": extracted_ngành_id,
                }
            )

        return candidates

    except Exception as e:
        print(f"⚠ Search Error (HTTP): {e}")
        return []



def search_legacy_fallback(
    collection_name: str, q_vec: np.ndarray, top_k: int
) -> List[Dict]:
    """
    Hỗ trợ code cũ trong lúc chờ migration.
    Hiện tại không dùng nữa → trả [].
    """
    return []


def trigger_config_reload():
    """
    Reload lại cấu hình collections_config từ SQLite
    """
    return get_collections_with_descriptions()
