
import torch
import os

FILE = "data.pth"

if not os.path.exists(FILE):
    print(f"File {FILE} không tồn tại!")
else:
    print(f"--- ĐANG ĐỌC FILE {FILE} ---")
    data = torch.load(FILE, map_location=torch.device('cpu'))
    
    print("\n1. CÁC THÔNG SỐ CẤU HÌNH (METADATA):")
    print(f"- input_size (Số lượng từ vựng): {data['input_size']}")
    print(f"- hidden_size (Số neuron lớp ẩn): {data['hidden_size']}")
    print(f"- output_size (Số lượng Category): {data['output_size']}")
    
    print("\n2. DANH SÁCH CATEGORY (TAGS):")
    print(data['tags'])
    
    print("\n3. TRỌNG SỐ MÔ HÌNH (MODEL STATE):")
    for key, tensor in data['model_state'].items():
        print(f"- {key}: kích thước {tensor.shape}")
        
    print("\n4. TỪ ĐIỂN (ALL WORDS - Một vài từ đầu tiên):")
    print(data['all_words'][:20]) # In 20 từ đầu tiên
    print(f"... và {len(data['all_words']) - 20} từ khác.")
