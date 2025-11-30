# chat_strict_no_hallucination.py
import sqlite3
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ==================== CẤU HÌNH ====================
FAQ_DB_PATH = "faq.db"
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen2:1.5b"

embed_model = SentenceTransformer("keepitreal/vietnamese-sbert")

# ==================== TẢI DỮ LIỆU 3 BẢNG ====================
print("🔄 Đang tải dữ liệu từ SQLite...")
conn = sqlite3.connect(FAQ_DB_PATH)
cursor = conn.cursor()

# Đọc 3 bảng
cursor.execute("SELECT question, answer, category FROM faq WHERE approved = 1 OR approved IS NULL")
FAQ_DATA = cursor.fetchall()

cursor.execute("""
    SELECT b.name, b.author, b.year, b.quantity, b.status, m.name
    FROM books b LEFT JOIN majors m ON b.major_id = m.major_id
""")
BOOKS_DATA = cursor.fetchall()

cursor.execute("SELECT name, major_id, description FROM majors")
MAJORS_DATA = cursor.fetchall()
conn.close()

print(f"✅ Đã tải: {len(FAQ_DATA)} FAQ, {len(BOOKS_DATA)} sách, {len(MAJORS_DATA)} ngành")

# ==================== BƯỚC 1: AI PHÂN LOẠI ====================
def route_question_with_ai(question: str) -> str:
    """Chỉ dùng Ollama để phân loại, không trả lời"""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if r.status_code != 200:
            return "UNKNOWN"
    except:
        return "UNKNOWN"
    
    prompt = f"""
Bạn là bộ phân loại câu hỏi. CHỈ phân loại, KHÔNG trả lời.

Phân loại câu hỏi thành ĐÚNG 1 loại:
- FAQ: quy định, giờ mở cửa, dịch vụ thư viện
- BOOKS: sách, tài liệu, tác giả  
- MAJORS: ngành học, mã ngành, đào tạo
- CHAT: chào hỏi, cảm ơn, tạm biệt
- UNKNOWN: không rõ ràng

Câu hỏi: "{question}"
Chỉ trả về 1 từ: FAQ, BOOKS, MAJORS, CHAT hoặc UNKNOWN.
"""

    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.0, "num_predict": 10}
        }, timeout=10)
        
        if r.status_code == 200:
            response = r.json().get("response", "").strip().upper()
            for category in ["FAQ", "BOOKS", "MAJORS", "CHAT", "UNKNOWN"]:
                if category in response:
                    return category
        return "UNKNOWN"
    except:
        return "UNKNOWN"

# ==================== BƯỚC 2: TÌM TRONG 3 BẢNG ====================
def search_in_faq(question: str) -> str:
    """Tìm trong FAQ - trả về answer nếu tìm thấy"""
    if not FAQ_DATA:
        return None
        
    query_vec = embed_model.encode(question, normalize_embeddings=True)
    best_similarity = 0
    best_answer = None
    
    for q, a, cat in FAQ_DATA:
        if q:
            q_vec = embed_model.encode(q, normalize_embeddings=True)
            similarity = np.dot(query_vec, q_vec)
            if similarity > best_similarity:
                best_similarity = similarity
                best_answer = a
    
    return best_answer if best_similarity > 0.7 else None

def search_in_books(question: str) -> str:
    """Tìm trong BOOKS - trả về thông tin sách nếu tìm thấy"""
    if not BOOKS_DATA:
        return None
        
    query_vec = embed_model.encode(question, normalize_embeddings=True)
    best_similarity = 0
    best_book = None
    
    for name, author, year, qty, status, major in BOOKS_DATA:
        book_text = f"{name} {author} {major or ''}"
        book_vec = embed_model.encode(book_text, normalize_embeddings=True)
        similarity = np.dot(query_vec, book_vec)
        if similarity > best_similarity:
            best_similarity = similarity
            best_book = (name, author, year, qty, status, major)
    
    if best_similarity > 0.6 and best_book:
        name, author, year, qty, status, major = best_book
        return f"Sách: {name}\nTác giả: {author}\nNăm: {year}\nSố lượng: {qty}\nTrạng thái: {status}\nNgành: {major or 'Không rõ'}"
    return None

def search_in_majors(question: str) -> str:
    """Tìm trong MAJORS - trả về thông tin ngành nếu tìm thấy"""
    if not MAJORS_DATA:
        return None
        
    query_vec = embed_model.encode(question, normalize_embeddings=True)
    best_similarity = 0
    best_major = None
    
    for name, major_id, description in MAJORS_DATA:
        major_text = f"{name} {major_id} {description or ''}"
        major_vec = embed_model.encode(major_text, normalize_embeddings=True)
        similarity = np.dot(query_vec, major_vec)
        if similarity > best_similarity:
            best_similarity = similarity
            best_major = (name, major_id, description)
    
    if best_similarity > 0.7 and best_major:
        name, major_id, description = best_major
        return f"Ngành: {name}\nMã ngành: {major_id}\nMô tả: {description or 'Đang cập nhật'}"
    return None

# ==================== BƯỚC 3: POLISH (CHỈ LÀM ĐẸP) ====================
def polish_answer(raw_answer: str, question: str) -> str:
    """Chỉ làm đẹp câu trả lời có sẵn, KHÔNG thêm thông tin"""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if r.status_code != 200:
            return raw_answer
    except:
        return raw_answer

    prompt = f"""
Bạn là trợ lý thư viện. VIẾT LẠI câu trả lời sau cho tự nhiên hơn, nhưng TUYỆT ĐỐI GIỮ NGUYÊN thông tin.

CÂU HỎI: {question}

THÔNG TIN GỐC (KHÔNG ĐƯỢC THAY ĐỔI):
{raw_answer}

YÊU CẦU:
- Giữ nguyên TẤT CẢ thông tin trong "THÔNG TIN GỐC"
- Chỉ thay đổi cách diễn đạt cho tự nhiên
- KHÔNG thêm bất kỳ thông tin nào khác
- Vẫn đảm bảo đầy đủ các chi tiết

Câu trả lời đã được viết lại:
"""

    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.3, "num_predict": 200}
        }, timeout=15)
        
        if r.status_code == 200:
            polished = r.json().get("response", "").strip()
            # Kiểm tra xem có giữ đủ thông tin không
            if polished and len(polished) > len(raw_answer) * 0.5:
                return polished
        return raw_answer
    except:
        return raw_answer

# ==================== HÀM CHÍNH - TUYỆT ĐỐI KHÔNG BỊA ====================
def process_message(question: str) -> str:
    """Xử lý câu hỏi - CHỈ dùng dữ liệu từ 3 bảng"""
    question = question.strip()
    if not question:
        return "Xin chào! Tôi có thể giúp gì cho bạn?"

    # BƯỚC 1: AI phân loại
    category = route_question_with_ai(question)
    print(f"[DEBUG] Phân loại: '{question}' -> {category}")

    # BƯỚC 2: Tìm trong 3 bảng
    raw_answer = None
    
    if category == "FAQ":
        raw_answer = search_in_faq(question)
    elif category == "BOOKS":
        raw_answer = search_in_books(question)  
    elif category == "MAJORS":
        raw_answer = search_in_majors(question)
    elif category == "CHAT":
        return "Xin chào! Tôi là trợ lý thư viện. Tôi có thể giúp bạn tìm thông tin về sách, ngành học hoặc quy định thư viện."

    # BƯỚC 3: Xử lý kết quả
    if raw_answer:
        # Có dữ liệu -> polish
        return polish_answer(raw_answer, question)
    else:
        # KHÔNG có dữ liệu -> không được bịa
        return "Hiện tại hệ thống chưa có thông tin chính xác về câu hỏi này."

# ==================== CHẠY CHATBOT ====================
if __name__ == "__main__":
    print("🤖 CHATBOT THƯ VIỆN - TUYỆT ĐỐI KHÔNG BỊA THÔNG TIN")
    print("=" * 50)
    print("💬 Hỏi về: sách, ngành học, quy định thư viện")
    print("   Bot CHỈ trả lời dựa trên dữ liệu có sẵn")
    print("   Không có dữ liệu -> 'chưa có thông tin chính xác'")
    print("\n(Gõ 'quit' để thoát)\n")
    
    while True:
        try:
            user_input = input("👤 Bạn: ").strip()
            if user_input.lower() in ['quit', 'exit', 'thoát']:
                print("🤖 Bot: Cảm ơn bạn! Hẹn gặp lại!")
                break
                
            if user_input:
                response = process_message(user_input)
                print(f"🤖 Bot: {response}\n")
                
        except KeyboardInterrupt:
            print("\n🤖 Bot: Tạm biệt!")
            break
        except Exception as e:
            print(f"🤖 Bot: Có lỗi xảy ra: {e}")