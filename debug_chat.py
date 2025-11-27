
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chat import process_message, embed_model, search_faq_candidates, rerank_with_llm, strict_answer, normalize

queries = [
    "Trang Facebook của thư viện tên gì ?",
    "Tổng số lượng tài liệu mà thư viện hiện có là bao nhiêu ?",
    "Phòng mấy tra cứu ở tầng nào ?",
    "Có thể mượn sách giáo trình trong thời gian bao lâu ?"
]

print("--- BẮT ĐẦU DEBUG ---")
for q in queries:
    print(f"\n\n==========================================")
    print(f"QUERY: {q}")
    print(f"==========================================")
    
    # 1. Router (Skip for now, assume FAQ/General)
    q_vec = embed_model.encode(normalize(q), normalize_embeddings=True)
    
    # 2. Search
    print("\n[SEARCH CANDIDATES]")
    candidates = search_faq_candidates(q_vec, top_k=10)
    for c in candidates:
        print(f"  - [{c['score']:.4f}] {c['answer'][:100]}... (Cat: {c['category']})")
        
    if not candidates:
        print("  -> NO CANDIDATES FOUND")
        continue

    # 3. Rerank
    print("\n[RERANK]")
    best_cand = rerank_with_llm(q, candidates)
    if best_cand:
        print(f"  -> SELECTED: {best_cand['answer']}")
        
        # 4. Strict Answer
        print("\n[STRICT ANSWER]")
        ans = strict_answer(q, best_cand['answer'])
        print(f"  -> OUTPUT: {ans}")
    else:
        print("  -> RERANK REJECTED ALL")

print("\n--- KẾT THÚC DEBUG ---")
