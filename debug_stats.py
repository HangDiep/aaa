
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chat import strict_answer

question = "Tổng số lượng tài liệu mà thư viện hiện có là bao nhiêu ?"
knowledge = "Tính đến 31/6/2024, có 146.612 bản tài liệu, phục vụ 36 ngành đào tạo."

print(f"Question: {question}")
print(f"Knowledge: {knowledge}")
print("-" * 20)

ans = strict_answer(question, knowledge)
print(f"Output: {ans}")
