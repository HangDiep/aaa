
import sys
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
print("Loading model...")
model = SentenceTransformer("keepitreal/vietnamese-sbert")

# Target Answer
target_answer = "- Phòng đọc sách tại chỗ: lầu 2\n- Phòng mượn sách: tầng 1\n- Phòng máy tra cứu: tầng trệt\n- Phòng đọc chung."
target_cat = "Vị trí kho"
target_text = f"{target_cat}: {target_answer}"

# Queries
queries = [
    "Phòng tra cứu nằm chỗ nào vậy?",
    "Phòng tra cứu thuộc khu nào?",
    "Phòng tra cứu được bố trí ở khu vực nào?",
    "Phòng tra cứu nằm tại tầng mấy?",
    "Đi tới phòng tra cứu bằng cách nào?"
]

print(f"\nTarget: {target_text}")
target_vec = model.encode(target_text, normalize_embeddings=True)

print("\n--- SCORES ---")
for q in queries:
    q_vec = model.encode(q, normalize_embeddings=True)
    score = np.dot(target_vec, q_vec)
    print(f"Query: '{q}' -> Score: {score:.4f}")
