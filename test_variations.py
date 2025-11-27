
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chat import process_message

queries = [
    "Phòng tra cứu nằm chỗ nào vậy?",
    "Phòng tra cứu đặt tại vị trí nào?"
]

print("--- BẮT ĐẦU TEST KHẢ NĂNG HIỂU NGỮ NGHĨA ---")
for q in queries:
    print(f"\nCâu hỏi: {q}")
    response = process_message(q)
    print(f"Bot trả lời: {response}")
print("\n--- KẾT THÚC TEST ---")
