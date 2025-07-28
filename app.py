import feedparser
from flask import Flask, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import gunicorn # Mặc dù không import trực tiếp, cần có trong requirements.txt

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Danh sách các nguồn RSS của bạn
RSS_FEEDS = [
    'https://vietstock.vn/830/chung-khoan/co-phieu.rss',
    'https://cafef.vn/thi-truong-chung-khoan.rss',
    'https://vietstock.vn/145/chung-khoan/y-kien-chuyen-gia.rss',
    'https://vietstock.vn/737/doanh-nghiep/hoat-dong-kinh-doanh.rss',
    'https://vietstock.vn/1328/dong-duong/thi-truong-chung-khoan.rss'
]

def get_news_and_vectors():
    """
    Hàm lấy tin tức từ các nguồn RSS và chuyển đổi tiêu đề thành vector TF-IDF.
    """
    all_news = []
    titles = []

    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries:
            # Thu thập các thông tin cần thiết
            news_item = {
                'title': entry.get('title', 'N/A'),
                'link': entry.get('link', '#'),
                'published': entry.get('published', 'N/A'),
                'summary': entry.get('summary', 'N/A'),
                'source': feed.feed.title
            }
            all_news.append(news_item)
            titles.append(entry.get('title', ''))

    # Kiểm tra xem có tin tức nào không
    if not titles:
        return []

    # Tạo vector từ tiêu đề tin tức bằng TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(titles)

    # Thêm vector vào mỗi mục tin tức
    for i, news_item in enumerate(all_news):
        # Chuyển vector thưa thành mảng numpy dày và chuyển thành list để có thể JSON hóa
        vector = tfidf_matrix[i].toarray().flatten().tolist()
        news_item['vector'] = vector

    return all_news

@app.route('/news')
def get_news():
    """
    API endpoint để lấy danh sách tin tức cùng với vector.
    """
    news_with_vectors = get_news_and_vectors()
    if not news_with_vectors:
        return jsonify({"error": "Không thể lấy tin tức"}), 500
    return jsonify(news_with_vectors)

@app.route('/')
def home():
    """
    Trang chủ đơn giản để kiểm tra API có hoạt động không.
    """
    return "API tin tức vector đang hoạt động! Truy cập /news để xem dữ liệu."

if __name__ == '__main__':
    # Chạy ứng dụng (chỉ dùng cho môi trường phát triển local)
    app.run(debug=True)
