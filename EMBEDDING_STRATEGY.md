# 🧠 Chiến lược Embedding trong push_to_qdrant.py

## 📊 **Câu trả lời ngắn gọn:**

### **Cho FAQ:**
- ✅ **Embed**: `category + answer` (KHÔNG embed question!)
- 📝 **Ví dụ:** `"Quy định: Thư viện mở cửa 7h-22h hàng ngày"`

### **Lưu vào Qdrant:**
- **Vector**: Embedding của `category + answer`
- **Payload**: `{question, answer, category, notion_id, last_updated}`

---

## 🔍 **Phân tích chi tiết:**

### **1. FAQ Collection (Dòng 235-236):**

```python
# Dòng 235-236
FAQ_DATA = [
    (
        row[0],  # notion_id
        normalize(f"{row[3] or ''}: {row[2] or ''}"),  # ← EMBED TEXT
        #           ↑ category      ↑ answer
        row[1],  # question (chỉ lưu vào payload, KHÔNG embed)
        row[2],  # answer
        row[3],  # category
        row[4]   # last_updated
    ) 
    for row in faq_rows if row[0] is not None
]
```

**Giải thích:**
- `row[3]` = `category` (ví dụ: "Quy định")
- `row[2]` = `answer` (ví dụ: "Thư viện mở cửa 7h-22h")
- **Embed text** = `"Quy định: Thư viện mở cửa 7h-22h"`
- **Question KHÔNG được embed**, chỉ lưu vào payload!

---

### **2. Tại sao embed `category + answer` thay vì `question`?**

#### **Lý do 1: Semantic Search hiệu quả hơn**

Khi user hỏi: **"Thư viện mở cửa mấy giờ?"**

**Cách 1 (Embed question - KHÔNG TỐT):**
```
User query: "Thư viện mở cửa mấy giờ?"
↓ Vector search
Tìm trong DB: "Giờ mở cửa của thư viện?"
→ Phải match chính xác câu hỏi → Khó!
```

**Cách 2 (Embed answer - TỐT HƠN):**
```
User query: "Thư viện mở cửa mấy giờ?"
↓ Vector search
Tìm trong DB: "Quy định: Thư viện mở cửa 7h-22h hàng ngày"
→ Match theo semantic (nghĩa) → Dễ hơn!
```

#### **Lý do 2: Người dùng hỏi theo nhiều cách khác nhau**

Cùng 1 câu trả lời, nhưng có thể hỏi:
- "Mấy giờ mở cửa?"
- "Giờ hoạt động?"
- "Thư viện mở lúc nào?"
- "Thời gian làm việc?"

→ Nếu embed **answer**, tất cả đều match được!

#### **Lý do 3: Category giúp phân loại**

Thêm `category` vào đầu giúp:
- Phân biệt context (Quy định vs Dịch vụ vs Cơ sở vật chất)
- Tăng độ chính xác khi search

---

### **3. BOOKS Collection (Dòng 246-247):**

```python
BOOK_DATA = [
    (
        row[0],  # notion_id
        normalize(f"sách {row[1]}. tác giả {row[2]}. ngành {row[6] or ''}"),
        #           ↑ name      ↑ author        ↑ major
        row[1],  # name
        row[2],  # author
        row[3],  # year
        row[4],  # quantity
        row[5],  # status
        row[6],  # major
        row[7]   # last_updated
    )
    for row in book_rows if row[0] is not None
]
```

**Embed text:** `"sách Python Programming. tác giả John Doe. ngành Công nghệ thông tin"`

**Tại sao?**
- User có thể hỏi: "Sách về Python"
- User có thể hỏi: "Sách của John Doe"
- User có thể hỏi: "Sách CNTT"
→ Embed tất cả thông tin quan trọng!

---

### **4. MAJORS Collection (Dòng 256-257):**

```python
MAJOR_DATA = [
    (
        row[0],  # notion_id
        normalize(f"ngành {row[1]}. mã {row[2]}. {row[3] or ''}"),
        #           ↑ name    ↑ major_id  ↑ description
        row[1],  # name
        row[2],  # major_id
        row[3]   # description
    )
    for row in major_rows if row[0] is not None
]
```

**Embed text:** `"ngành Công nghệ thông tin. mã 7480201. Đào tạo kỹ sư CNTT..."`

---

## 🎯 **Luồng hoạt động khi User hỏi:**

### **Bước 1: User hỏi**
```
"Thư viện mở cửa mấy giờ?"
```

### **Bước 2: Chatbot tạo embedding cho câu hỏi**
```python
# chat.py
q_vec = embed_model.encode("thư viện mở cửa mấy giờ", normalize_embeddings=True)
```

### **Bước 3: Vector search trong Qdrant**
```python
# Tìm vector gần nhất với q_vec
results = qdrant_client.query_points(
    collection_name="faq",
    query=q_vec.tolist(),
    limit=10
)
```

### **Bước 4: Qdrant so sánh với các vectors đã lưu**
```
q_vec (câu hỏi user)
  ↓ Cosine similarity
  ↓
Vector 1: "Quy định: Thư viện mở cửa 7h-22h" → Score: 0.85 ✅
Vector 2: "Dịch vụ: Photocopy, in ấn"       → Score: 0.32
Vector 3: "Quy định: Mượn sách tối đa 5 quyển" → Score: 0.41
```

### **Bước 5: Lấy payload của vector có score cao nhất**
```json
{
  "question": "Giờ mở cửa của thư viện?",
  "answer": "Thư viện mở cửa 7h-22h hàng ngày",
  "category": "Quy định",
  "notion_id": "abc-123"
}
```

### **Bước 6: Chatbot trả lời**
```
"Thư viện mở cửa 7h-22h hàng ngày"
```

---

## 💡 **Ưu điểm của chiến lược này:**

### ✅ **1. Linh hoạt với cách hỏi khác nhau**
```
User: "Mấy giờ mở cửa?"
User: "Giờ hoạt động?"
User: "Thời gian làm việc?"
→ Tất cả đều match với "Thư viện mở cửa 7h-22h"
```

### ✅ **2. Không cần câu hỏi mẫu chính xác**
```
DB: "Giờ mở cửa của thư viện?"  ← Không cần lưu
Chỉ cần: "Thư viện mở cửa 7h-22h" ← Embed cái này
→ User hỏi bất kỳ cách nào cũng match!
```

### ✅ **3. Tận dụng semantic search**
```
User: "Khi nào thư viện hoạt động?"
→ "hoạt động" ≈ "mở cửa" (semantic similarity)
→ Match được!
```

### ✅ **4. Category giúp phân loại**
```
"Quy định: Mở cửa 7h-22h"
"Dịch vụ: Photocopy, in ấn"
→ Dễ phân biệt context
```

---

## ⚠️ **Lưu ý quan trọng:**

### **1. Question vẫn được lưu trong payload**
```python
payload = {
    "question": row[2] or "",  # ← Vẫn lưu!
    "answer": row[3] or "",
    "category": row[4] or "",
}
```
→ Dùng để hiển thị hoặc rerank sau khi search

### **2. Normalize text trước khi embed**
```python
def normalize(x: str) -> str:
    return " ".join(x.lower().strip().split())
```
→ Loại bỏ khoảng trắng thừa, lowercase

### **3. Embedding model: BAAI/bge-m3**
- Hỗ trợ tiếng Việt tốt
- Vector size: 1024 dimensions
- Fallback: `keepitreal/vietnamese-sbert`

---

## 📝 **Tóm tắt:**

| Collection | Embed gì? | Ví dụ |
|------------|-----------|-------|
| **FAQ** | `category + answer` | `"Quy định: Thư viện mở cửa 7h-22h"` |
| **BOOKS** | `name + author + major` | `"sách Python. tác giả John. ngành CNTT"` |
| **MAJORS** | `name + major_id + description` | `"ngành CNTT. mã 7480201. Đào tạo..."` |

**Question KHÔNG được embed**, chỉ lưu vào payload để hiển thị!

---

## 🎯 **Kết luận:**

Chiến lược này **THÔNG MINH** vì:
1. ✅ Linh hoạt với nhiều cách hỏi
2. ✅ Tận dụng semantic search
3. ✅ Không cần câu hỏi mẫu chính xác
4. ✅ Category giúp phân loại context

→ **Chatbot hiểu được ý nghĩa**, không chỉ match từ khóa! 🧠
