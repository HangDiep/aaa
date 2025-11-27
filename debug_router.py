from chat import route_llm, embed_model, normalize

queries = [
    "sách trí tuệ nhân tạo thì sao",
    "có sách về python không",
    "tìm giáo trình toán cao cấp",
    "mượn sách ở đâu",
    "quy trình trả sách"
]

print("--- Debugging Router ---")
for q in queries:
    q_vec = embed_model.encode(normalize(q), normalize_embeddings=True)
    route = route_llm(q, q_vec)
    print(f"Query: '{q}' -> Route: {route}")
