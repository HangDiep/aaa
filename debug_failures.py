
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chat import process_message, embed_model, search_faq_candidates, rerank_with_llm, strict_answer, normalize, route_llm

queries = [
    "Hướng dẫn trả sách tại thư viện",
    "Quy trình trả sách",
    "bị hư hoặc mất thì phải bồi thường không ?",
    "Chức năng của thư viện gồm những gì ?",
    "Quản lý tài liệu gồm những gì?",
    "Tôi lỡ làm rách tài liệu thì có bị sao không?"
]

print("--- BẮT ĐẦU DEBUG QUẢN LÝ & BỔ SUNG ---")
for q in queries:
    print(f"\n\n==========================================")
    print(f"QUERY: {q}")
    print(f"==========================================")
    
    q_vec = embed_model.encode(normalize(q), normalize_embeddings=True)
    
    # 1. Route
    intent = route_llm(q, q_vec)
    print(f"INTENT: {intent}")

    # 2. Search
    print("\n[SEARCH CANDIDATES]")
    candidates = search_faq_candidates(q_vec, top_k=10, filter_category=intent if intent in ["Nhiệm vụ", "Chức năng"] else None)
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
        print("\n[STRICT ANSWER INPUT]")
        print(f"  Question: {q}")
        print(f"  Knowledge: {best_cand['answer']}")
        
        ans = strict_answer(q, best_cand['answer'])
        print(f"  -> OUTPUT: {ans}")
    else:
        print("  -> RERANK REJECTED ALL")

print("\n--- KẾT THÚC DEBUG ---")
