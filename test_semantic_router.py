from chat import process_message, route_llm, embed_model, normalize

# Force load anchors if not already loaded (chat.py loads them on import but let's be sure)
# In chat.py, compute_anchor_vectors() is called at module level, so it should be ready.

queries = [
    "Tìm tài liệu về AI",             # Expect: BOOKS
    "Tổng số lượng tài liệu",         # Expect: FAQ
    "Giờ mở cửa thư viện",            # Expect: FAQ
    "Ngành Công nghệ thông tin",      # Expect: MAJORS
    "Sách Mắt Biếc còn không",        # Expect: BOOKS
    "Quy trình mượn trả sách",        # Expect: FAQ
    "Phòng đọc sách tại chỗ ở đâu",   # Expect: FAQ
    "Phòng máy tra cứu ở tầng mấy",   # Expect: FAQ
    "Khuôn viên trường có cây xi không", # Expect: FAQ (Implicit Fallback)
    "Canteen bán đồ ăn gì",           # Expect: FAQ (Implicit Fallback)
    "Thời tiết hôm nay thế nào"       # Expect: FAQ (Implicit Fallback)
]

print("--- Testing Semantic Router ---")
for q in queries:
    print(f"\nQuery: '{q}'")
    # Embed query manually to test route_llm directly
    q_vec = embed_model.encode(normalize(q), normalize_embeddings=True)
    route = route_llm(q, q_vec)
    print(f"-> Final Route: {route}")
