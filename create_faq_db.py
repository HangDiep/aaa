import sqlite3

# Đường dẫn đến file DB (tạo trong thư mục hiện tại)
DB_PATH = "faq.db"

# Kết nối và tạo DB nếu chưa tồn tại
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Tạo table FAQ nếu chưa tồn tại
cur.execute("""
CREATE TABLE IF NOT EXISTS faq (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    category TEXT,
    language TEXT DEFAULT 'Tiếng Việt',
    approved INTEGER DEFAULT 1,  -- 1: Đã duyệt (Yes), 0: Chưa duyệt
    last_updated TEXT
);
""")

# Dữ liệu FAQ từ Notion (chuyển đổi ngày "14 tháng 9, 2025" thành "2025-09-14")
faq_data = [
    ("Địa chỉ thư viện ở đâu?", "567 Lê Duẩn, Buôn Ma Thuột, Đắk Lắk", "Địa chỉ", "Tiếng Việt", 1, "2025-09-14"),
    ("Số điện thoại liên hệ?", "0262.3825180", "Liên hệ", "Tiếng Việt", 1, "2025-09-14"),
    ("Email liên hệ?", "thuvien@ttn.edu.vn", "Liên hệ", "Tiếng Việt", 1, "2025-09-14"),
    ("Trang Facebook của thư viện?", "https://www.facebook.com/thuviendhtn", "Liên hệ", "Tiếng Việt", 1, "2025-09-14"),
    ("Giờ mở cửa thư viện?", "Sáng: 7h30 - 11h30, Chiều: 13h30 - 17h00 (theo giờ hành chính Nhà trường).", "Giờ mở cửa", "Tiếng Việt", 1, "2025-09-14"),
    ("Số lượng tài liệu trong thư viện?", "Tính đến 31/6/2024, có 146.612 bản tài liệu, phục vụ 36 ngành đào tạo.", "Nguồn lực", "Tiếng Việt", 1, "2025-09-14"),
    ("Vị trí các phòng trong thư viện?", "- Phòng đọc sách tại chỗ: lầu 2\n- Phòng mượn sách: tầng trệt\n- Phòng máy tra cứu: tầng trệt\n- Phòng đọc chung.", "Vị trí kho", "Tiếng Việt", 1, "2025-09-14"),
    ("Nội quy mượn sách giáo trình?", "Mượn theo môn học, không hạn chế số lượng, thời gian trong 5 tháng.", "Quy định", "Tiếng Việt", 1, "2025-09-14"),
    ("Nội quy mượn sách tham khảo?", "Mượn tối đa 3 quyển, thời gian 15 ngày.", "Quy định", "Tiếng Việt", 1, "2025-09-14"),
    ("Thủ tục mượn sách?", "Trình thẻ, tìm tài liệu, CB quét máy, ghi vào sổ mượn do CB cấp.", "Quy định", "Tiếng Việt", 1, "2025-09-14"),
    ("Thủ tục trả sách?", "Trình thẻ và tài liệu, CB quét trả và ký vào sổ nhận.", "Quy định", "Tiếng Việt", 1, "2025-09-14"),
    ("Phạt khi quá hạn mượn sách?", "500đ/ngày/1 cuốn.", "Quy định", "Tiếng Việt", 1, "2025-09-14"),
    ("Xử lý khi mất hoặc hư hỏng tài liệu?", "Mua đền nếu có, nếu không thì đền gấp 2 lần giá thị trường.", "Quy định", "Tiếng Việt", 1, "2025-09-14"),
    ("Chức năng chính của thư viện?", "Quản lý và tổ chức hoạt động phục vụ đào tạo, nghiên cứu khoa học.", "Chức năng", "Tiếng Việt", 1, "2025-09-14"),
    ("Nhiệm vụ bổ sung tài liệu?", "Thu nhận tài liệu từ Trường, lập kế hoạch bổ sung theo yêu cầu các đơn vị.", "Nhiệm vụ", "Tiếng Việt", 1, "2025-09-14"),
    ("Nhiệm vụ quản lý tài liệu?", "Xử lý, sắp xếp, xây dựng hệ thống tra cứu, phục vụ và hướng dẫn khai thác tài liệu.", "Nhiệm vụ", "Tiếng Việt", 1, "2025-09-14"),
    ("Nhiệm vụ khác của thư viện?", "Liên kết với các thư viện khác, bồi dưỡng chuyên môn, đảm bảo chất lượng.", "Nhiệm vụ", "Tiếng Việt", 1, "2025-09-14"),
    ("Ai là giám đốc thư viện?", "ThS. Vũ Thị Giang", "Lãnh đạo", "Tiếng Việt", 1, "2025-09-14"),
    ("Ai phụ trách kho đọc?", "ThS. Vũ Đình Trung", "Cơ cấu tổ chức", "Tiếng Việt", 1, "2025-09-14"),
    ("Nội quy phòng đọc sách?", "Phải xuất trình thẻ sinh viên/thẻ bạn đọc. Trang phục lịch sự. Chỉ đọc tại chỗ, trả tài liệu sau khi đọc, có thể photocopy nếu cần.", "Quy định", "Tiếng Việt", 1, "2025-09-14")
]

# Thêm dữ liệu vào table
cur.executemany("""
INSERT INTO faq (question, answer, category, language, approved, last_updated)
VALUES (?, ?, ?, ?, ?, ?);
""", faq_data)

# Commit và đóng kết nối
conn.commit()
conn.close()

print("DB FAQ đã được tạo thành công tại:", DB_PATH)
