
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chat import process_message, embed_model, search_faq_candidates, rerank_with_llm, strict_answer, normalize

queries = [
    "Thư viện gồm những phòng nào ?",
    "Thư viện có những nhiệm vụ nào ?",
    "Ngoài quản lý tài liệu thì thư viện còn những nhiệm vụ gì ?",
    "Quản lý tài liệu gồm những gì?"
]

print("--- BẮT ĐẦU DEBUG NHIỆM VỤ & PHÒNG ---")
for q in queries:
    print(f"\n\n==========================================")
    print(f"QUERY: {q}")
    print(f"==========================================")
    
    q_vec = embed_model.encode(normalize(q), normalize_embeddings=True)
    
    # Search with top_k=15
    print("\n[SEARCH CANDIDATES]")
    candidates = search_faq_candidates(q_vec, top_k=15)
    for c in candidates:
        print(f"  - [{c['score']:.4f}] {c['answer'][:100]}... (Cat: {c['category']})")
        
    if not candidates:
        print("  -> NO CANDIDATES FOUND")
        continue

    # Rerank
    print("\n[RERANK]")
    best_cand = rerank_with_llm(q, candidates)
    if best_cand:
        print(f"  -> SELECTED: {best_cand['answer']}")
        
        # Strict Answer
        print("\n[STRICT ANSWER]")
        ans = strict_answer(q, best_cand['answer'])
        print(f"  -> OUTPUT: {ans}")
    else:
        print("  -> RERANK REJECTED ALL")

print("\n--- KẾT THÚC DEBUG ---")
