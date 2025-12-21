# ==========================================
# HO TÊN: Đỗ Thị Hồng Điệp
# MSSV: 23103014
# ĐỒ ÁN: Chatbot Dynamic Router - TTN University
# NGÀY NỘP: 21/12/2025
# Copyright © 2025. All rights reserved.
# ==========================================


import os
import torch
import easyocr
from pathlib import Path
from typing import Optional, List

# ---------- Khởi tạo OCR Reader (chỉ 1 lần) ----------
print("[OCR] Đang khởi tạo EasyOCR (chỉ 1 lần duy nhất)...")
OCR_READER = easyocr.Reader(['vi', 'en'], gpu=torch.cuda.is_available())
print("[OCR] Khởi tạo EasyOCR thành công!")

# ---------- Hàm quét văn bản từ ảnh ----------
def ocr_from_image(image_path: str) -> Optional[str]:
    """
    Quét văn bản từ hình ảnh, trả về chuỗi kết quả.
    Nếu không đọc được, trả về None.
    
    Args:
        image_path (str): Đường dẫn tới file ảnh.
    
    Returns:
        Optional[str]: Văn bản quét được hoặc None.
    """
    if not image_path or not Path(image_path).exists():
        print(f"[OCR] File không tồn tại: {image_path}")
        return None
    
    try:
        print(f"[OCR] Đang quét ảnh: {image_path}")
        results: List[str] = OCR_READER.readtext(str(image_path), detail=0, paragraph=True)
        text = "\n".join(results).strip()
        if text:
            print(f"[OCR] Thành công – {len(text)} ký tự")
            print("[OCR TEXT]:", text)   # 👈 THÊM DÒNG NÀY – in toàn bộ text
            return text
        else:
            print(f"[OCR] Không tìm thấy văn bản trong ảnh: {image_path}")
            return None
    except Exception as e:
        print(f"[OCR] Lỗi khi quét ảnh '{image_path}': {e}")
        return None

# ---------- Test nhanh khi chạy file trực tiếp ----------
if __name__ == "__main__":
    test_image = "test.jpg"  # đổi tên file theo hình ảnh thử nghiệm của bạn
    text = ocr_from_image(test_image)
    if text:
        print("Kết quả OCR:\n", text)
    else:
        print("Không đọc được văn bản từ ảnh.")
