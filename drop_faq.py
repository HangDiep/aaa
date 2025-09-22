import sqlite3

conn = sqlite3.connect("faq.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS faq;")

conn.commit()
conn.close()

print("Đã xóa bảng faq thành công!")
