import torch

path = "data.pth"

data = torch.load(path, map_location="cpu")

print("Loại dữ liệu:", type(data))
print("Các key trong file:", list(data.keys()))

# In FULL ALL_WORDS
all_words = data.get("all_words", [])
print("\n===== FULL TỪ ĐIỂN (ALL WORDS) =====")
for i, w in enumerate(all_words):
    print(f"{i+1}. {w}")
print("===== TỔNG SỐ TỪ:", len(all_words), "=====")

# In TAGS
tags = data.get("tags", [])
print("\n===== CÁC CATEGORY (TAGS) =====")
for t in tags:
    print("-", t)
print("===== TỔNG SỐ TAG:", len(tags), "=====")

# In info model
model_state = data.get("model_state", {})
print("\n===== TRỌNG SỐ MÔ HÌNH =====")
for k, v in model_state.items():
    print(f"{k}: {v.shape}")
