# Library Chatbot - TTN University
## 🤖 Hệ thống Chatbot Thông minh Tự động Đồng bộ Notion

Hệ thống chatbot hỗ trợ tra cứu thông tin thư viện với khả năng tự động nhận diện ý định, hỗ trợ đọc văn bản từ ảnh (OCR) và nhận diện giọng nói (Voice-to-Text). Đặc biệt, hệ thống có khả năng tự động đồng bộ dữ liệu động từ bất kỳ Database Notion nào mà bạn chia sẻ.

---

## 📂 1. Cấu trúc Thư mục Dự án

```text
Chat_bot/
├── view/                   # 🎨 Giao diện & Main Server
│   ├── app.py              # 🚀 File chạy chính (FastAPI + WebSocket)
│   ├── index.html          # Landing page
│   ├── Chatbot.html        # Giao diện chat trực quan
│   ├── app.js              # Logic frontend (Chat & Voice)
│   └── chatbot.css         # Styling cho giao diện
├── rag/                    # 🧠 Cấu hình RAG & Vector Store
│   └── .env                # Lưu API Key (Notion, Qdrant, LLM)
├── banghiamcuoicung/      # 🎤 Module Xử lý Giọng nói
│   └── server.py           # WebSocket Router cho Whisper model
├── sync_dynamic.py         # 🔄 Đồng bộ Notion -> SQLite
├── push_to_qdrant_dynamic.py # ⤴️ Lập chỉ mục SQLite -> Qdrant
├── chat_fixed.py           # ⚙️ Logic xử lý hội thoại & SQLite
├── chat_dynamic_router.py  # �️ Routing câu hỏi đến đúng Collection
├── ocr_helper.py           # 📸 Trích xuất văn bản từ hình ảnh
├── faq.db / chat.db        # 📊 Cơ sở dữ liệu SQLite local
└── requirements.txt        # 📦 Danh sách thư viện cần thiết
```

---

## 🛠️ 2. Hướng dẫn Cài đặt

### Bước 1: Thiết lập Môi trường ảo
```bash
python -m venv venv
source venv/bin/activate  # Trên Linux/Mac
# hoặc
.\venv\Scripts\Activate.ps1 # Trên Windows (Powershell)
```

### Bước 2: Cài đặt Thư viện
```bash
pip install -r requirements.txt
```

### Bước 3: Cấu hình .env
Tạo file `.env` trong thư mục `rag/` với nội dung:
```env
NOTION_API_KEY=your_notion_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
ZIPUR_API_KEY=your_llm_api_key
```

---

## � 3. Vận hành Hệ thống

Thực hiện theo đúng thứ tự sau để dữ liệu được sẵn sàng:

1.  **Đồng bộ Dữ liệu**:
    Chạy script để lấy dữ liệu từ Notion về SQLite:
    ```bash
    python sync_dynamic.py
    ```
2.  **Lập chỉ mục Vector**:
    Đẩy dữ liệu từ SQLite lên Qdrant để tìm kiếm thông minh:
    ```bash
    python push_to_qdrant_dynamic.py [tên_bảng]
    ```
3.  **Khởi chạy Server**:
    Chạy giao diện web và chatbot tích hợp:
    ```bash
    python -m uvicorn view.app:app --reload
    ```
    Truy cập: `http://127.0.0.1:8000`

---

## ✨ 4. Các Tính năng Chính

*   **Dynamic Routing**: Chatbot tự động phát hiện bạn đang hỏi về sách, ngành học hay quy định để truy vấn đúng bảng dữ liệu.
*   **Hỗ trợ OCR**: Tải ảnh lên (ảnh chụp trang sách, thông báo), chatbot sẽ tự đọc chữ và trả lời dựa trên nội dung đó.
*   **Hỗ trợ Giọng nói**: Nhấn nút mic để nói, hệ thống sử dụng model Whisper (tiny) để chuyển đổi tiếng Việt cực nhanh.
*   **Tự động Discovery**: Chỉ cần bạn "Share" một Database mới trên Notion cho Integration, hệ thống sẽ tự phát hiện và tạo bảng tương ứng.

---
