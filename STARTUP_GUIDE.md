# 🚀 Hướng dẫn khởi động Chatbot (Không cần mở nhiều terminal!)

## 🎯 **3 Cách khởi động:**

---

## **Cách 1: Tự động khởi động (KHUYẾN NGHỊ)** ⭐

### **Bước 1: Chạy script**
```
Double-click: start_all.bat
```

### **Kết quả:**
- ✅ Mở 2 cửa sổ terminal riêng biệt
- ✅ Terminal 1: Ngrok (hiển thị URL)
- ✅ Terminal 2: Server (hiển thị logs)
- ✅ Bạn có thể minimize các cửa sổ này

### **Ưu điểm:**
- ✅ Chỉ cần double-click 1 file
- ✅ Xem được logs của cả 2 services
- ✅ Dễ debug khi có lỗi

### **Nhược điểm:**
- ⚠️ Có 2 cửa sổ terminal (nhưng có thể minimize)

---

## **Cách 2: Chạy ngầm (Background)** 🔇

### **Bước 1: Chạy script**
```
Double-click: start_background.bat
```

### **Kết quả:**
- ✅ Chạy ngầm, không hiển thị terminal
- ✅ Sạch sẽ, không có cửa sổ làm phiền

### **Xem ngrok URL:**
```
Double-click: check_ngrok_url.bat
```
Hoặc mở browser: `http://localhost:4040`

### **Dừng services:**
```
Double-click: stop_all.bat
```

### **Ưu điểm:**
- ✅ Không có cửa sổ terminal
- ✅ Chạy ngầm như service

### **Nhược điểm:**
- ⚠️ Không thấy logs (khó debug)
- ⚠️ Phải mở browser để xem ngrok URL

---

## **Cách 3: Dùng Windows Task Scheduler (Tự động khi khởi động máy)** 🤖

### **Bước 1: Mở Task Scheduler**
```
Win + R → taskschd.msc → Enter
```

### **Bước 2: Tạo Task mới**
1. Click "Create Basic Task"
2. Name: "Chatbot Auto Start"
3. Trigger: "When I log on"
4. Action: "Start a program"
5. Program: `D:\HTML\a - Copy\start_background.bat`
6. Finish

### **Kết quả:**
- ✅ Tự động chạy khi bật máy
- ✅ Không cần nhớ phải chạy

### **Ưu điểm:**
- ✅ Hoàn toàn tự động
- ✅ Như một Windows Service

### **Nhược điểm:**
- ⚠️ Luôn chạy khi bật máy (tốn RAM)
- ⚠️ Khó debug

---

## 📋 **So sánh 3 cách:**

| Tiêu chí | Cách 1 (start_all.bat) | Cách 2 (start_background.bat) | Cách 3 (Task Scheduler) |
|----------|------------------------|--------------------------------|-------------------------|
| **Dễ dùng** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Xem logs** | ✅ Dễ | ❌ Khó | ❌ Khó |
| **Tự động** | ❌ Phải chạy tay | ❌ Phải chạy tay | ✅ Tự động |
| **Sạch sẽ** | ⚠️ Có 2 cửa sổ | ✅ Không có cửa sổ | ✅ Không có cửa sổ |
| **Debug** | ✅ Dễ | ❌ Khó | ❌ Khó |

---

## 🎯 **Khuyến nghị:**

### **Đang phát triển/debug:**
→ Dùng **Cách 1** (start_all.bat)

### **Đã ổn định, dùng hàng ngày:**
→ Dùng **Cách 2** (start_background.bat)

### **Muốn tự động khi bật máy:**
→ Dùng **Cách 3** (Task Scheduler)

---

## 📝 **Các file script đã tạo:**

| File | Mô tả |
|------|-------|
| `start_all.bat` | Khởi động cả 2 services (hiển thị terminal) |
| `start_background.bat` | Khởi động ngầm (không hiển thị terminal) |
| `stop_all.bat` | Dừng tất cả services |
| `check_ngrok_url.bat` | Xem ngrok URL (mở browser) |

---

## 🔧 **Cách sử dụng:**

### **Khởi động hàng ngày:**
```
1. Double-click: start_all.bat (hoặc start_background.bat)
2. Đợi 3-5 giây
3. Double-click: check_ngrok_url.bat (để lấy URL)
4. Copy URL vào n8n (nếu URL thay đổi)
5. Bắt đầu dùng chatbot!
```

### **Khi kết thúc:**
```
Double-click: stop_all.bat
```

---

## 💡 **Tips:**

### **1. Tạo shortcut trên Desktop:**
- Right-click `start_all.bat` → Send to → Desktop (create shortcut)
- Đổi tên: "🚀 Start Chatbot"
- Đổi icon: Right-click → Properties → Change Icon

### **2. Pin vào Taskbar:**
- Tạo shortcut như trên
- Right-click shortcut → Pin to taskbar

### **3. Xem ngrok URL nhanh:**
Mở browser: `http://localhost:4040`

### **4. Kiểm tra services đang chạy:**
```
Task Manager (Ctrl+Shift+Esc)
→ Tìm "ngrok.exe" và "python.exe"
```

---

## 🆘 **Troubleshooting:**

### **Lỗi: "ngrok not found"**
→ Cài ngrok và thêm vào PATH:
```
1. Download: https://ngrok.com/download
2. Giải nén vào C:\ngrok\
3. Thêm C:\ngrok\ vào System PATH
```

### **Lỗi: "Port 8000 already in use"**
→ Dừng process đang dùng port 8000:
```
netstat -ano | findstr :8000
taskkill /F /PID <PID>
```

### **Ngrok URL thay đổi mỗi lần chạy**
→ Đăng ký ngrok account (free) để có URL cố định:
```
1. Đăng ký tại: https://dashboard.ngrok.com/signup
2. Copy authtoken
3. Chạy: ngrok config add-authtoken <YOUR_TOKEN>
4. Sửa start_all.bat: ngrok http 8000 --domain=<YOUR_DOMAIN>
```

---

## ✅ **Kết luận:**

**Không cần mở nhiều terminal nữa!** Chỉ cần:

1. **Double-click `start_all.bat`** → Mọi thứ tự động chạy
2. **Minimize 2 cửa sổ** (hoặc dùng `start_background.bat`)
3. **Bắt đầu dùng chatbot!**

🎉 **Đơn giản và tiện lợi!**
