# 🔥 TÌM THẤY VẤN ĐỀ!

## ❌ Vấn đề:

```
Ngrok:  http://5000  → https://mallory-hydrated-sophie.ngrok-free.dev
Server: http://8000  ← Uvicorn đang chạy ở đây
```

**Ngrok forward sai port!** Nên là:
```
Ngrok:  http://8000  → https://mallory-hydrated-sophie.ngrok-free.dev
Server: http://8000  ✅
```

---

## ✅ Giải pháp:

### **Cách 1: Restart ngrok với port đúng (KHUYẾN NGHỊ)**

1. **Dừng ngrok hiện tại:**
   ```bash
   # Ctrl+C trong terminal đang chạy ngrok
   ```

2. **Chạy lại với port 8000:**
   ```bash
   ngrok http 8000
   ```

3. **Copy URL mới** (có thể khác với URL cũ)

4. **Cập nhật URL trong n8n** với URL mới

---

### **Cách 2: Chạy uvicorn trên port 5000**

```bash
# Dừng uvicorn hiện tại (Ctrl+C)
# Chạy lại với port 5000:
uv run uvicorn chat_fixed:app --host 0.0.0.0 --port 5000 --workers 1
```

---

## 🎯 Khuyến nghị:

**Dùng Cách 1** vì:
- Port 8000 là default của uvicorn
- Dễ nhớ và chuẩn
- Không cần thay đổi code

---

## 📋 Sau khi fix:

1. **Test ngrok:**
   ```bash
   curl https://mallory-hydrated-sophie.ngrok-free.dev/notion/faq
   ```

2. **Trigger workflow từ Notion**

3. **Kiểm tra terminal server** → Sẽ thấy log request

---

## 💡 Tip:

Để tránh nhầm lẫn, luôn check:
```bash
# Terminal 1: Ngrok
ngrok http 8000

# Terminal 2: Server
uv run uvicorn chat_fixed:app --workers 1
# (mặc định port 8000)
```

Hoặc dùng **ngrok config** để fix port:
```yaml
# ~/.ngrok2/ngrok.yml
tunnels:
  chatbot:
    proto: http
    addr: 8000
```

Rồi chạy:
```bash
ngrok start chatbot
```
