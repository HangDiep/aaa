import sqlite3

conn = sqlite3.connect("faq.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS majors;")

conn.commit()
conn.close()

print("Đã xóa bảng majors thành công!")
