from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
import feedparser
import requests
from typing import List

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="News Vector API",
    description="API để lấy tin tức từ các nguồn RSS và chuyển đổi thành vector.",
    version="1.0.0",
)

# Tải một mô hình nhẹ đã được tối ưu hóa.
# 'all-MiniLM-L6-v2' là một mô hình tốt, cân bằng giữa tốc độ, kích thước và chất lượng.
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    # Xử lý lỗi nếu không tải được mô hình
    model = None
    print(f"Lỗi khi tải mô hình SentenceTransformer: {e}")


# Danh sách các RSS feed của bạn
RSS_FEEDS = [
    "https://vietstock.vn/830/chung-khoan/co-phieu.rss",
    "https://cafef.vn/thi-truong-chung-khoan.rss",
    "https://vietstock.vn/145/chung-khoan/y-kien-chuyen-gia.rss",
    "https://vietstock.vn/737/doanh-nghiep/hoat-dong-kinh-doanh.rss",
,"https://vietstock.vn/1328/dong-duong/thi-truong-chung-khoan.rss",
]

# Cache đơn giản để lưu trữ tin tức đã lấy
# Giúp giảm số lần gọi đến RSS feed, tăng tốc độ phản hồi
news_cache = {}

def fetch_news_from_feed(feed_url: str) -> List[dict]:
    """Hàm lấy và phân tích tin tức từ một RSS feed."""
    try:
        # Sử dụng ETag và Last-Modified để kiểm tra xem feed có được cập nhật không
        headers = {}
        if feed_url in news_cache:
            if "etag" in news_cache[feed_url]:
                headers['If-None-Match'] = news_cache[feed_url]['etag']
            if "modified" in news_cache[feed_url]:
                headers['If-Modified-Since'] = news_cache[feed_url]['modified']

        response = requests.get(feed_url, headers=headers, timeout=10)

        # Nếu feed không thay đổi, trả về dữ liệu từ cache
        if response.status_code == 304:
            return news_cache[feed_url]['entries']

        # Phân tích cú pháp feed
        parsed_feed = feedparser.parse(response.content)

        # Cập nhật cache
        news_cache[feed_url] = {
            'entries': parsed_feed.entries,
            'etag': response.headers.get('ETag'),
            'modified': response.headers.get('Last-Modified')
        }
        return parsed_feed.entries

    except requests.RequestException as e:
        print(f"Lỗi khi lấy tin từ {feed_url}: {e}")
        return []

@app.get("/news-vectors/", summary="Lấy tin tức dưới dạng vector")
async def get_news_vectors():
    """
    Endpoint này lấy tin tức mới nhất từ tất cả các nguồn RSS,
    chuyển đổi tiêu đề và mô tả của chúng thành vector nhúng (embeddings).
    """
    if not model:
        raise HTTPException(status_code=503, detail="Mô hình NLP hiện không khả dụng.")

    all_news = []
    for feed_url in RSS_FEEDS:
        entries = fetch_news_from_feed(feed_url)
        for entry in entries:
            # Chỉ lấy các thông tin cần thiết
            news_item = {
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "summary": entry.get("summary", ""),
                "published": entry.get("published", "")
            }
            all_news.append(news_item)

    if not all_news:
        return {"message": "Không có tin tức nào được tìm thấy.", "vectors": []}

    # Tạo nội dung để chuyển đổi thành vector (kết hợp tiêu đề và tóm tắt)
    texts_to_embed = [f"{news['title']}. {news['summary']}" for news in all_news]

    # Chuyển đổi thành vector
    embeddings = model.encode(texts_to_embed, convert_to_tensor=False).tolist()

    # Kết hợp thông tin tin tức với vector tương ứng
    vectorized_news = []
    for i, news_item in enumerate(all_news):
        news_item['vector'] = embeddings[i]
        vectorized_news.append(news_item)

    return {"news": vectorized_news}

@app.get("/", summary="Endpoint kiểm tra trạng thái")
def read_root():
    """Endpoint gốc để kiểm tra xem API có hoạt động không."""
    return {"status": "API đang hoạt động"}

