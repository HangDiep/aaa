# import sqlite3
# from flask import Flask, request, jsonify

# app = Flask(__name__)

# def get_db_connection():
#     conn = sqlite3.connect('faq.db')
#     conn.row_factory = sqlite3.Row  # Để trả về dict thay vì tuple
#     return conn

# @app.route('/inventory', methods=['GET'])
# def check_inventory():
#     book_name = request.args.get('book_name')
#     if not book_name:
#         return jsonify({'error': 'book_name là bắt buộc'}), 400

#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM books WHERE name LIKE ?", ('%' + book_name + '%',))
#     books = cursor.fetchall()
#     conn.close()

#     if not books:
#         return jsonify({'message': 'Không tìm thấy sách'}), 404

#     result = []
#     for book in books:
#         result.append({
#             'name': book['name'],
#             'author': book['author'],
#             'year': book['year'],
#             'quantity': book['quantity'],
#             'status': book['status'],
#             'major_id': book['major_id'],
#             'link': book['link'],
#             'available': book['available']
#         })

#     return jsonify(result)

# if __name__ == '__main__':
#     app.run(debug=True)