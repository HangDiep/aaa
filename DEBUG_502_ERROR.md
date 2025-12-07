# 🔍 Hướng dẫn Debug lỗi 502 Bad Gateway từ n8n

## 📊 Tình trạng hiện tại:

✅ **Endpoint `/notion/faq` HOẠT ĐỘNG BÌNH THƯỜNG** khi test từ Python  
❌ **n8n gửi request → Lỗi 502 Bad Gateway**

---

## 🎯 Nguyên nhân có thể:

### 1. **n8n gửi data sai format**
   - Thiếu field bắt buộc (`notion_id`, `question`, `answer`)
   - Kiểu dữ liệu sai (ví dụ: `approved` phải là `int`, không phải `string`)
   - Field name sai (ví dụ: `Question` thay vì `question`)

### 2. **Ngrok timeout**
   - Request từ n8n → ngrok → localhost:8000
   - Nếu server xử lý quá lâu (>30s), ngrok sẽ trả 502

### 3. **Server crash khi nhận request**
   - Lỗi trong code xử lý
   - Exception không được catch

---

## 🛠️ Các bước debug:

### **Bước 1: Restart server với code mới**

Tôi đã thêm debug endpoint vào `sync_n8n_to_sqlite.py` (đúng kiến trúc). Hãy restart server:

```bash
# Dừng server hiện tại (Ctrl+C trong terminal đang chạy uvicorn)
# Chạy lại:
uv run uvicorn chat_fixed:app --workers 1
```

---

### **Bước 2: Thay đổi URL trong n8n**

Trong n8n workflow, **TẠM THỜI** thay đổi URL từ:
```
https://mallory-hydrated-sophie.ngrok-free.dev/notion/faq
```

Thành:
```
https://mallory-hydrated-sophie.ngrok-free.dev/notion/debug/faq
```

**Lưu ý:** Endpoint debug nằm trong router `/notion`, nên URL đầy đủ là `/notion/debug/faq`

---

### **Bước 3: Trigger workflow từ Notion**

1. Vào Notion database "Faq_API"
2. Sửa 1 record bất kỳ (ví dụ: thêm dấu cách vào Answer)
3. Tick/untick checkbox "Approved"
4. Xem terminal server → sẽ in ra **TOÀN BỘ** data n8n gửi

---

### **Bước 4: Phân tích output**

Terminal sẽ hiển thị:

```
================================================================================
🔍 DEBUG /debug/notion/faq
================================================================================

📋 Headers:
   host: 127.0.0.1:8000
   user-agent: axios/1.6.0
   content-type: application/json
   ...

📦 Raw Body (XXX bytes):
{
  "notion_id": "...",
  "question": "...",
  ...
}

🔧 Parsed JSON:
{
  "notion_id": "2a5db606-cea8-8122-bdaa-fed10d1e5ef0",
  "question": "Test question",
  "answer": "Test answer",
  ...
}
================================================================================
```

---

### **Bước 5: So sánh với format đúng**

Format đúng phải có:

```json
{
  "notion_id": "string",     // ✅ Bắt buộc
  "question": "string",      // ✅ Bắt buộc
  "answer": "string",        // ✅ Bắt buộc
  "category": "string",      // ⚠️ Optional
  "language": "vi",          // ⚠️ Optional (default: "vi")
  "approved": 1              // ⚠️ Optional (default: 1), phải là NUMBER
}
```

**Lỗi thường gặp:**
- ❌ `"approved": "1"` (string thay vì number)
- ❌ `"Question": "..."` (viết hoa Q)
- ❌ Thiếu `notion_id`

---

## 🔧 Sửa lỗi trong n8n:

### **Nếu thiếu field:**

Trong n8n HTTP Request node, thêm field vào Body:

```javascript
{
  "notion_id": "{{ $json.id }}",
  "question": "{{ $json.properties.Question.rich_text[0].plain_text }}",
  "answer": "{{ $json.properties.Answer.rich_text[0].plain_text }}",
  "category": "{{ $json.properties.Category.select?.name || null }}",
  "language": "vi",
  "approved": 1  // ✅ Phải là number, không có dấu ngoặc kép
}
```

### **Nếu field name sai:**

Đảm bảo tên field **viết thường** và **khớp** với Pydantic model:

```python
class FAQItem(BaseModel):
    notion_id: str      # ✅ Phải là "notion_id", không phải "notionId" hay "notion_ID"
    question: str       # ✅ Phải là "question", không phải "Question"
    answer: str         # ✅ Phải là "answer", không phải "Answer"
    category: Optional[str] = None
    language: Optional[str] = "vi"
    approved: Optional[int] = 1  # ✅ Phải là int (1), không phải string ("1")
```

---

## 📸 Screenshot debug output

Sau khi chạy Bước 3, **chụp màn hình terminal** và gửi cho tôi để phân tích chi tiết.

---

## ✅ Sau khi fix xong:

1. **Đổi lại URL** trong n8n từ `/debug/notion/faq` → `/notion/faq`
2. **Test lại** bằng cách sửa record trong Notion
3. **Kiểm tra database**:

```bash
python test_notion_endpoint.py
```

Hoặc:

```sql
-- Mở faq.db bằng SQLite browser
SELECT * FROM faq ORDER BY last_updated DESC LIMIT 5;
```

---

## 🆘 Nếu vẫn lỗi:

Gửi cho tôi:
1. Screenshot terminal khi chạy debug endpoint
2. Screenshot n8n workflow (HTTP Request node configuration)
3. Log lỗi từ terminal server

---

## 💡 Tip:

Nếu muốn test nhanh mà không cần n8n, dùng **curl**:

```bash
curl -X POST http://127.0.0.1:8000/notion/faq \
  -H "Content-Type: application/json" \
  -d '{
    "notion_id": "test-curl-123",
    "question": "Test từ curl",
    "answer": "Đây là test",
    "category": "Test",
    "language": "vi",
    "approved": 1
  }'
```

Hoặc dùng **Postman** / **Insomnia** để test.
