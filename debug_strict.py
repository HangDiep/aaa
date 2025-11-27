
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chat import process_message, embed_model, search_faq_candidates, rerank_with_llm, strict_answer, normalize

queries = [
    "Làm sao để trả sách ?",
    "Thủ tục trả sách như thế nào ?",
    "Quên trả sách đúng hạn thì phải làm sao ?",
    "Có bị phạt khi trễ hạn trả sách không ?",
    "Ngoài quản lý tài liệu thì thư viện còn những nhiệm vụ gì ?",
    "Tôi lỡ làm rách tài liệu thì có bị sao không?"
]

print("--- BẮT ĐẦU DEBUG STRICT ANSWER ---")
for q in queries:
    print(f"\n\n==========================================")
    print(f"QUERY: {q}")
    print(f"==========================================")
    
    q_vec = embed_model.encode(normalize(q), normalize_embeddings=True)
    
    # Search
    candidates = search_faq_candidates(q_vec, top_k=15)
    if not candidates:
        print("  -> NO CANDIDATES FOUND")
        continue

    # Rerank
    best_cand = rerank_with_llm(q, candidates)
    if best_cand:
        print(f"  -> RERANK SELECTED: {best_cand['answer']}")
        
        # Strict Answer
        print("\n[STRICT ANSWER INPUT]")
        print(f"  Question: {q}")
        print(f"  Knowledge: {best_cand['answer']}")
        
        ans = strict_answer(q, best_cand['answer'])
        print(f"  -> OUTPUT: {ans}")
    else:
        print("  -> RERANK REJECTED ALL")

print("\n--- KẾT THÚC DEBUG ---")
