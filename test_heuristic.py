from chat import process_message

queries = [
    "tài liệu trí tuệ nhân tạo",
    "tìm ấn phẩm về python",
    "giáo trình toán cao cấp"
]

print("--- Testing Heuristic with Synonyms ---")
for q in queries:
    print(f"\nQuery: '{q}'")
    response = process_message(q)
    print(f"Response: {response}")
