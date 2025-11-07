import sqlite3

conn = sqlite3.connect("faq.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS books;")

conn.commit()
conn.close()

print("Đã xóa bảng books thành công!")
