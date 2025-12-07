# ✅ CHECKLIST: Công việc bạn cần làm

## 🔴 **QUAN TRỌNG - CHƯA LÀM:**

### **1. Fix ngrok chạy sai port** ❌
```bash
# Hiện tại:
ngrok http 5000  ❌ SAI!

# Cần sửa thành:
ngrok http 8000  ✅ ĐÚNG!
```

**Cách làm:**
1. Ctrl+C trong terminal đang chạy ngrok
2. Chạy lại: `ngrok http 8000`
3. Copy URL mới (có thể khác URL cũ)
4. Cập nhật URL trong n8n workflow

**Tại sao quan trọng?**
- Server đang chạy port 8000
- Ngrok đang forward port 5000
- → Request từ n8n không đến được server → 502 Bad Gateway

---

### **2. Cập nhật URL trong n8n workflow** ❌

Sau khi restart ngrok, URL có thể thay đổi. Cần:

1. Vào n8n workflow
2. Tìm HTTP Request node
3. Cập nhật URL mới từ ngrok
4. **Đảm bảo endpoint đúng:** `/notion/faq` (KHÔNG phải `/debug/notion/faq`)

---

### **3. Restart server để load code mới** ⚠️

Tôi đã thêm code tự động push lên Qdrant trong `sync_n8n_to_sqlite.py`. Cần restart:

```bash
# Ctrl+C trong terminal đang chạy uvicorn
# Chạy lại:
uv run uvicorn chat_fixed:app --host 0.0.0.0 --port 8000 --workers 1
```

---

### **4. Test luồng hoàn chỉnh** ❌

Sau khi fix ngrok và restart server, cần test:

**Test 1: Endpoint hoạt động qua ngrok**
```bash
# Thay URL bằng URL ngrok của bạn
curl -X POST https://YOUR-NGROK-URL.ngrok-free.dev/notion/faq \
  -H "Content-Type: application/json" \
  -d '{
    "notion_id": "test-ngrok-123",
    "question": "Test qua ngrok",
    "answer": "Đây là test",
    "approved": 1
  }'
```

**Test 2: n8n workflow**
1. Vào Notion database "Faq_API"
2. Sửa 1 record bất kỳ (thêm dấu cách vào Answer)
3. Giữ nguyên tick Approved=✅
4. Xem terminal server → Phải thấy log:
   ```
   📥 Received FAQ data: {...}
   ✅ Inserted/Updated FAQ: ...
   🔄 Đang push lên Qdrant...
   ✅ Qdrant sync started (background)
   ```

**Test 3: Kiểm tra Qdrant đã nhận data**
```bash
python test_notion_endpoint.py
```

**Test 4: Chatbot học được câu mới**
1. Hỏi chatbot câu đã sửa trong Notion
2. Xem có trả lời đúng không

---

## 🟡 **NÊN LÀM (Tùy chọn):**

### **5. Cấu hình ngrok cố định port** (Tùy chọn)

Để tránh nhầm lẫn sau này, tạo file config:

```yaml
# File: ~/.ngrok2/ngrok.yml (hoặc C:\Users\Admin\.ngrok2\ngrok.yml trên Windows)

tunnels:
  chatbot:
    proto: http
    addr: 8000
```

Sau đó chạy:
```bash
ngrok start chatbot
```

---

### **6. Tạo script tự động khởi động** (Tùy chọn)

Tạo file `start.bat`:

```batch
@echo off
echo Starting chatbot services...

start "Ngrok" cmd /k "ngrok http 8000"
timeout /t 3

start "Server" cmd /k "uv run uvicorn chat_fixed:app --host 0.0.0.0 --port 8000 --workers 1"

echo All services started!
pause
```

Chạy `start.bat` để khởi động tất cả cùng lúc.

---

### **7. Kiểm tra Qdrant dashboard** (Tùy chọn)

Mở browser:
```
http://localhost:6333/dashboard
```

Xem collection "faq" có bao nhiêu records.

---

## ✅ **ĐÃ LÀM XONG:**

- ✅ Code tự động push lên Qdrant (đã thêm vào `sync_n8n_to_sqlite.py`)
- ✅ Debug endpoint (đã thêm `/notion/debug/faq`)
- ✅ Test script (`test_notion_endpoint.py`)
- ✅ Tài liệu hướng dẫn (COMPLETE_FLOW.md, EMBEDDING_STRATEGY.md, etc.)

---

## 📋 **TÓM TẮT CÔNG VIỆC CẦN LÀM NGAY:**

### **Bước 1: Fix ngrok**
```bash
# Terminal 1
ngrok http 8000
```

### **Bước 2: Restart server**
```bash
# Terminal 2
uv run uvicorn chat_fixed:app --host 0.0.0.0 --port 8000 --workers 1
```

### **Bước 3: Cập nhật URL trong n8n**
- Copy URL mới từ ngrok
- Paste vào n8n HTTP Request node
- Endpoint: `/notion/faq`

### **Bước 4: Test**
1. Sửa record trong Notion
2. Xem terminal server có log không
3. Kiểm tra database có data mới không
4. Test chatbot có học được không

---

## 🎯 **Ưu tiên:**

1. **NGAY LẬP TỨC:** Fix ngrok (Bước 1-3)
2. **SAU ĐÓ:** Test luồng hoàn chỉnh (Bước 4)
3. **TÙY CHỌN:** Các bước 5-7 (làm sau khi mọi thứ hoạt động)

---

## 🆘 **Nếu gặp vấn đề:**

- **502 Bad Gateway:** Kiểm tra lại ngrok port
- **Không thấy log:** Kiểm tra n8n workflow URL
- **Chatbot không học:** Chờ 5-10s để Qdrant sync xong
- **Khác:** Gửi screenshot terminal cho tôi

---

**Bạn cần làm GẤP nhất là Bước 1-3!** 🚀
