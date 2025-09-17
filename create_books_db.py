import sqlite3

# Kết nối đến DB (sử dụng faq.db để tích hợp chung)
conn = sqlite3.connect('faq.db')
c = conn.cursor()

# Tạo bảng majors nếu chưa tồn tại
c.execute('''
    CREATE TABLE IF NOT EXISTS majors (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT
    )
''')

# Tạo bảng books nếu chưa tồn tại
c.execute('''
    CREATE TABLE IF NOT EXISTS books (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        author TEXT,
        year INTEGER,
        quantity INTEGER,
        major_id INTEGER,
        status TEXT,
        link TEXT,
        available TEXT,
        FOREIGN KEY (major_id) REFERENCES majors(id)
    )
''')

# Chèn dữ liệu majors (dựa trên dữ liệu bạn cung cấp)
majors = [
    (1, 'Công nghệ thông tin', None),
    (2, 'Kinh tế', None),
    (3, 'Y học', None),
    (4, 'Giáo dục Mầm non', None),
    (5, 'Giáo dục Tiểu học', None),
    (6, 'Giáo dục Tiểu học - Tiếng Jrai', None),
    (7, 'Giáo dục Chính trị', None),
    (8, 'Sư phạm Toán học', None),
    (9, 'Sư phạm Vật lý', None),
    (10, 'Sư phạm Hóa học', None)
]
for major in majors:
    c.execute("INSERT OR IGNORE INTO majors (id, name, description) VALUES (?, ?, ?)", major)

# Chèn dữ liệu books (dựa trên dữ liệu bạn cung cấp, tôi liệt kê đầy đủ)
books = [
    ('Cấu trúc dữ liệu & Giải thuật', 'Nguyễn Văn A', 2020, 12, 1, 'Còn hàng', 'https://www.notion.so/26f57231d19d80739d01ec90bd35c24b', 'Yes'),
    ('Trí tuệ nhân tạo', 'Trần Văn B', 2021, 8, 1, 'Còn hàng', 'https://www.notion.so/26f57231d19d80739d01ec90bd35c24b', 'Yes'),
    ('Lập trình Python', 'Lê Văn C', 2019, 15, 1, 'Còn hàng', 'https://www.notion.so/26f57231d19d80739d01ec90bd35c24b', 'Yes'),
    ('An toàn thông tin', 'Nguyễn Văn D', 2022, 10, 1, 'Còn hàng', 'https://www.notion.so/26f57231d19d80739d01ec90bd35c24b', 'Yes'),
    ('Hệ điều hành', 'Phạm Văn E', 2018, 6, 1, 'Còn hàng', 'https://www.notion.so/26f57231d19d80739d01ec90bd35c24b', 'Yes'),
    ('Mạng máy tính', 'Hoàng Văn F', 2019, 14, 1, 'Còn hàng', 'https://www.notion.so/26f57231d19d80739d01ec90bd35c24b', 'Yes'),
    ('Lập trình Web', 'Nguyễn Văn G', 2021, 9, 1, 'Còn hàng', 'https://www.notion.so/26f57231d19d80739d01ec90bd35c24b', 'Yes'),
    ('Khoa học dữ liệu', 'Trần Văn H', 2020, 13, 1, 'Còn hàng', 'https://www.notion.so/26f57231d19d80739d01ec90bd35c24b', 'Yes'),
    ('Machine Learning', 'Phạm Văn I', 2022, 7, 1, 'Còn hàng', 'https://www.notion.so/26f57231d19d80739d01ec90bd35c24b', 'Yes'),
    ('Cơ sở dữ liệu', 'Nguyễn Văn K', 2019, 11, 1, 'Còn hàng', 'https://www.notion.so/26f57231d19d80739d01ec90bd35c24b', 'Yes'),
    ('Kinh tế học vi mô', 'Nguyễn Thị A', 2018, 10, 2, 'Còn hàng', 'https://www.notion.so/26f57231d19d802499bdd0b48ec77f6c', 'Yes'),
    ('Kinh tế học vĩ mô', 'Trần Thị B', 2019, 8, 2, 'Còn hàng', 'https://www.notion.so/26f57231d19d802499bdd0b48ec77f6c', 'Yes'),
    ('Marketing căn bản', 'Lê Thị C', 2020, 14, 2, 'Còn hàng', 'https://www.notion.so/26f57231d19d802499bdd0b48ec77f6c', 'Yes'),
    ('Quản trị kinh doanh', 'Nguyễn Văn D', 2021, 7, 2, 'Còn hàng', 'https://www.notion.so/26f57231d19d802499bdd0b48ec77f6c', 'Yes'),
    ('Tài chính doanh nghiệp', 'Phạm Văn E', 2019, 15, 2, 'Còn hàng', 'https://www.notion.so/26f57231d19d802499bdd0b48ec77f6c', 'Yes'),
    ('Kế toán tài chính', 'Hoàng Văn F', 2018, 6, 2, 'Còn hàng', 'https://www.notion.so/26f57231d19d802499bdd0b48ec77f6c', 'Yes'),
    ('Phân tích đầu tư', 'Nguyễn Văn G', 2022, 12, 2, 'Còn hàng', 'https://www.notion.so/26f57231d19d802499bdd0b48ec77f6c', 'Yes'),
    ('Quản trị nhân sự', 'Trần Văn H', 2021, 9, 2, 'Còn hàng', 'https://www.notion.so/26f57231d19d802499bdd0b48ec77f6c', 'Yes'),
    ('Thương mại điện tử', 'Phạm Văn I', 2020, 11, 2, 'Còn hàng', 'https://www.notion.so/26f57231d19d802499bdd0b48ec77f6c', 'Yes'),
    ('Kinh doanh quốc tế', 'Nguyễn Văn K', 2022, 13, 2, 'Còn hàng', 'https://www.notion.so/26f57231d19d802499bdd0b48ec77f6c', 'Yes'),
    ('Giải phẫu học', 'Nguyễn Thị A', 2018, 12, 3, 'Còn hàng', 'https://www.notion.so/26f57231d19d80ce9c7cdf794eb8ecb4', 'Yes'),
    ('Sinh lý học', 'Trần Thị B', 2019, 8, 3, 'Còn hàng', 'https://www.notion.so/26f57231d19d80ce9c7cdf794eb8ecb4', 'Yes'),
    ('Dược lý học', 'Lê Thị C', 2020, 14, 3, 'Còn hàng', 'https://www.notion.so/26f57231d19d80ce9c7cdf794eb8ecb4', 'Yes'),
    ('Chẩn đoán hình ảnh', 'Nguyễn Văn D', 2021, 7, 3, 'Còn hàng', 'https://www.notion.so/26f57231d19d80ce9c7cdf794eb8ecb4', 'Yes'),
    ('Nội khoa cơ bản', 'Phạm Văn E', 2019, 10, 3, 'Còn hàng', 'https://www.notion.so/26f57231d19d80ce9c7cdf794eb8ecb4', 'Yes'),
    ('Ngoại khoa cơ bản', 'Hoàng Văn F', 2018, 6, 3, 'Còn hàng', 'https://www.notion.so/26f57231d19d80ce9c7cdf794eb8ecb4', 'Yes'),
    ('Sản phụ khoa', 'Nguyễn Văn G', 2022, 15, 3, 'Còn hàng', 'https://www.notion.so/26f57231d19d80ce9c7cdf794eb8ecb4', 'Yes'),
    ('Nhi khoa', 'Trần Văn H', 2021, 9, 3, 'Còn hàng', 'https://www.notion.so/26f57231d19d80ce9c7cdf794eb8ecb4', 'Yes'),
    ('Y học cổ truyền', 'Phạm Văn I', 2020, 11, 3, 'Còn hàng', 'https://www.notion.so/26f57231d19d80ce9c7cdf794eb8ecb4', 'Yes'),
    ('Dinh dưỡng học', 'Nguyễn Văn K', 2022, 13, 3, 'Còn hàng', 'https://www.notion.so/26f57231d19d80ce9c7cdf794eb8ecb4', 'Yes'),
    ('Phương pháp tổ chức hoạt động vui chơi cho trẻ mầm non', 'Nguyễn Thị Lan', 2018, 12, 4, 'Còn hàng', 'https://www.notion.so/26f57231d19d80288b89c79f52019b8b', 'Yes'),
    ('Tâm lý học trẻ em lứa tuổi mầm non', 'Trần Văn Minh', 2019, 9, 4, 'Còn hàng', 'https://www.notion.so/26f57231d19d80288b89c79f52019b8b', 'Yes'),
    ('Giáo dục âm nhạc cho trẻ mầm non', 'Lê Thị Hoa', 2020, 8, 4, 'Còn hàng', 'https://www.notion.so/26f57231d19d80288b89c79f52019b8b', 'Yes'),
    ('Phát triển ngôn ngữ cho trẻ mầm non', 'Phạm Thị Hương', 2017, 10, 4, 'Còn hàng', 'https://www.notion.so/26f57231d19d80288b89c79f52019b8b', 'Yes'),
    ('Giáo dục thể chất cho trẻ mầm non', 'Nguyễn Văn Thành', 2018, 6, 4, 'Còn hàng', 'https://www.notion.so/26f57231d19d80288b89c79f52019b8b', 'Yes'),
    ('Chăm sóc sức khỏe trẻ mầm non', 'Nguyễn Thị Thu', 2021, 11, 4, 'Còn hàng', 'https://www.notion.so/26f57231d19d80288b89c79f52019b8b', 'Yes'),
    ('Tổ chức hoạt động tạo hình cho trẻ mầm non', 'Vũ Thị Mai', 2019, 5, 4, 'Còn hàng', 'https://www.notion.so/26f57231d19d80288b89c79f52019b8b', 'Yes'),
    ('Phát triển tình cảm – kỹ năng xã hội cho trẻ mầm non', 'Lê Văn Dũng', 2020, 13, 4, 'Còn hàng', 'https://www.notion.so/26f57231d19d80288b89c79f52019b8b', 'Yes'),
    ('Giáo dục lễ giáo cho trẻ mầm non', 'Hoàng Thị Thanh', 2022, 14, 4, 'Còn hàng', 'https://www.notion.so/26f57231d19d80288b89c79f52019b8b', 'Yes'),
    ('Tổ chức hoạt động khám phá khoa học cho trẻ mầm non', 'Nguyễn Thị Oanh', 2021, 20, 4, 'Còn hàng', 'https://www.notion.so/26f57231d19d80288b89c79f52019b8b', 'Yes'),
    ('Phương pháp dạy học Toán ở tiểu học', 'Nguyễn Văn An', 2018, 8, 5, 'Còn hàng', 'https://www.notion.so/26f57231d19d8008b35fd01c6693924a', 'Yes'),
    ('Phương pháp dạy học Tiếng Việt ở tiểu học', 'Trần Thị Bình', 2019, 9, 5, 'Còn hàng', 'https://www.notion.so/26f57231d19d8008b35fd01c6693924a', 'Yes'),
    ('Giáo dục đạo đức ở tiểu học', 'Lê Văn Cường', 2020, 7, 5, 'Còn hàng', 'https://www.notion.so/26f57231d19d8008b35fd01c6693924a', 'Yes'),
    ('Phương pháp dạy học Tự nhiên và Xã hội', 'Nguyễn Thị Hạnh', 2017, 11, 5, 'Còn hàng', 'https://www.notion.so/26f57231d19d8008b35fd01c6693924a', 'Yes'),
    ('Tâm lý học lứa tuổi tiểu học', 'Phạm Thị Mai', 2018, 10, 5, 'Còn hàng', 'https://www.notion.so/26f57231d19d8008b35fd01c6693924a', 'Yes'),
    ('Đánh giá kết quả học tập ở tiểu học', 'Nguyễn Văn Hoàng', 2021, 6, 5, 'Còn hàng', 'https://www.notion.so/26f57231d19d8008b35fd01c6693924a', 'Yes'),
    ('Phát triển kỹ năng sống cho học sinh tiểu học', 'Vũ Thị Thu', 2019, 15, 5, 'Còn hàng', 'https://www.notion.so/26f57231d19d8008b35fd01c6693924a', 'Yes'),
    ('Ứng dụng CNTT trong dạy học tiểu học', 'Lê Thị Thanh', 2022, 5, 5, 'Còn hàng', 'https://www.notion.so/26f57231d19d8008b35fd01c6693924a', 'Yes'),
    ('Giáo dục thể chất ở tiểu học', 'Nguyễn Văn Tuấn', 2020, 13, 5, 'Còn hàng', 'https://www.notion.so/26f57231d19d8008b35fd01c6693924a', 'Yes'),
    ('Phương pháp dạy học Âm nhạc ở tiểu học', 'Hoàng Thị Nga', 2021, 17, 5, 'Còn hàng', 'https://www.notion.so/26f57231d19d8008b35fd01c6693924a', 'Yes')
]
for book in books:
    c.execute("INSERT INTO books (name, author, year, quantity, major_id, status, link, available) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", book)

# Commit và đóng kết nối
conn.commit()
conn.close()

print("DB đã được tạo và dữ liệu đã được chèn thành công!")