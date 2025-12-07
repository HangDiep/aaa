# 🐚 Shell Scripts vs Batch Files

## 📊 **Sự khác biệt:**

| Đặc điểm | `.bat` (Batch) | `.sh` (Shell) |
|----------|----------------|---------------|
| **Hệ điều hành** | Windows | Linux/Mac/WSL |
| **Shell** | CMD / PowerShell | Bash / Zsh |
| **Cú pháp** | DOS commands | Unix commands |
| **Chạy bằng** | Double-click | `./script.sh` hoặc `bash script.sh` |

---

## 🎯 **Bạn nên dùng gì?**

### **Nếu dùng Windows thuần:**
→ Dùng **`.bat`** files
```
Double-click: start_all.bat
```

### **Nếu dùng Git Bash trên Windows:**
→ Dùng **`.sh`** files
```bash
chmod +x start_all.sh
./start_all.sh
```

### **Nếu dùng WSL (Windows Subsystem for Linux):**
→ Dùng **`.sh`** files
```bash
chmod +x start_all.sh
./start_all.sh
```

---

## 📁 **Files đã tạo:**

### **Batch files (Windows CMD):**
- `start_all.bat`
- `start_background.bat`
- `stop_all.bat`
- `check_ngrok_url.bat`

### **Shell scripts (Git Bash/WSL):**
- `start_all.sh` ✨ MỚI
- `stop_all.sh` ✨ MỚI
- `check_ngrok_url.sh` ✨ MỚI

---

## 🚀 **Cách dùng Shell Scripts:**

### **Bước 1: Cho phép thực thi (chỉ làm 1 lần)**
```bash
chmod +x start_all.sh
chmod +x stop_all.sh
chmod +x check_ngrok_url.sh
```

### **Bước 2: Chạy script**
```bash
./start_all.sh
```

### **Hoặc:**
```bash
bash start_all.sh
```

---

## 💡 **Tại sao người ta hay dùng `.sh`?**

### **1. Cross-platform (Đa nền tảng)**
- ✅ Chạy được trên Linux
- ✅ Chạy được trên Mac
- ✅ Chạy được trên Windows (qua Git Bash/WSL)
- ❌ `.bat` chỉ chạy trên Windows

### **2. Powerful (Mạnh mẽ hơn)**
- ✅ Bash có nhiều tính năng hơn CMD
- ✅ Dễ xử lý text, pipes, conditions
- ✅ Cú pháp chuẩn Unix

### **3. Professional (Chuyên nghiệp)**
- ✅ Dùng trong DevOps, CI/CD
- ✅ Dùng trong Docker, Kubernetes
- ✅ Dùng trong production servers

### **4. Version Control (Quản lý phiên bản)**
- ✅ Git xử lý line endings tốt hơn với `.sh`
- ⚠️ `.bat` có thể bị lỗi line endings (CRLF vs LF)

---

## 🎯 **Khuyến nghị:**

### **Nếu bạn là developer:**
→ Dùng **`.sh`** với **Git Bash**
- Cài Git for Windows: https://git-scm.com/download/win
- Mở Git Bash
- Chạy: `./start_all.sh`

### **Nếu bạn chỉ dùng Windows:**
→ Dùng **`.bat`**
- Đơn giản hơn
- Double-click là chạy

---

## 🔧 **So sánh cú pháp:**

### **Batch (.bat):**
```batch
@echo off
echo Starting services...
start "Ngrok" cmd /k "ngrok http 8000"
timeout /t 3
```

### **Shell (.sh):**
```bash
#!/bin/bash
echo "Starting services..."
ngrok http 8000 &
sleep 3
```

→ Shell script **ngắn gọn và rõ ràng hơn**!

---

## ✅ **Kết luận:**

**Cả 2 đều OK!** Tùy vào môi trường:

- **Windows CMD** → Dùng `.bat`
- **Git Bash / WSL / Linux / Mac** → Dùng `.sh`

**Tôi đã tạo cả 2 loại cho bạn rồi!** Chọn loại nào phù hợp với bạn nhé! 🎉

---

## 🆘 **Troubleshooting:**

### **Lỗi: "Permission denied" khi chạy .sh**
```bash
chmod +x start_all.sh
```

### **Lỗi: "bad interpreter" hoặc "^M: not found"**
→ Line endings sai (CRLF thay vì LF)

**Fix:**
```bash
dos2unix start_all.sh
# Hoặc
sed -i 's/\r$//' start_all.sh
```

### **Lỗi: "ngrok: command not found"**
→ Thêm ngrok vào PATH:
```bash
export PATH=$PATH:/c/ngrok
# Thêm vào ~/.bashrc để permanent
```
