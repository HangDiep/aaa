import os
from chat import rerank_with_llm

# Mock candidates based on user logs and data
candidates = [
    {"answer": "các phòng thư viện: - Phòng đọc sách tại chỗ: lầu 2 - Phòng mượn sách: tầng trệt - Phòng máy tra cứu: tầng trệt - Phòng đọc chung: ở tầng 3", "category": "Vị trí kho"},
    {"answer": "Mượn sách:Trình thẻ, tìm tài liệu, CB quét máy, ghi vào sổ mượn do CB cấp.", "category": "Quy định"},
    {"answer": "Trả sách: Trình thẻ và tài liệu, CB quét trả và ký vào sổ nhận.", "category": "Quy định"},
    {"answer": "Nhiệm vụ khác của thư viện: Liên kết với các thư viện khác, bồi dưỡng chuyên môn, đảm bảo chất lượng.", "category": "Nhiệm vụ"},
    {"answer": "Nhiệm vụ quản lý tài liệu: Xử lý, sắp xếp, xây dựng hệ thống tra cứu, phục vụ và hướng dẫn khai thác tài liệu.", "category": "Nhiệm vụ"},
    {"answer": "Mượn giáo trình: Mượn theo môn học, không hạn chế số lượng, thời gian trong 5 tháng.", "category": "Quy định"},
]

queries = [
    "Phòng mượn sách ở đâu?",
    "Hướng dẫn trả sách tại thư viện",
    "Ngoài quản lý tài liệu thì thư viện còn những nhiệm vụ gì ?",
    "Sách giáo trình có thể mượn theo môn học không ?"
]

print("--- TESTING RERANK ---")
for q in queries:
    print(f"\nQuery: {q}")
    result = rerank_with_llm(q, candidates)
    if result:
        print(f"✅ Selected: {result['answer']}")
    else:
        print("❌ No selection")
