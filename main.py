# main.py
import uvicorn
import feedparser
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

# --- Cấu hình ---
# Cấu hình logging để dễ dàng gỡ lỗi trên Render
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Vietnam News Vector API",
    description="API để lấy tin tức chứng khoán từ RSS, trích xuất nội dung và chuyển thành vector.",
    version="1.0.0",
)

# Danh sách các nguồn RSS
RSS_FEEDS = [
    "https://vietstock.vn/830/chung-khoan/co-phieu.rss",
    "https://cafef.vn/thi-truong-chung-khoan.rss",
    "https://vietstock.vn/145/chung-khoan/y-kien-chuyen-gia.rss",
    "https://vietstock.vn/737/doanh-nghiep/hoat-dong-kinh-doanh.rss",
    "https://vietstock.vn/1328/dong-duong/thi-truong-chung-khoan.rss",
]

# Tải mô hình để vector hóa. Mô hình này hỗ trợ tốt tiếng Việt.
# Lần đầu chạy sẽ mất chút thời gian để tải mô hình về.
logger.info("Đang tải mô hình Sentence Transformer...")
model = SentenceTransformer('keepitreal/vietnamese-sbert')
logger.info("Tải mô hình thành công.")

# Khởi tạo công cụ chia nhỏ văn bản
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Số ký tự mỗi đoạn
    chunk_overlap=50  # Số ký tự chồng lấn giữa các đoạn
)

# --- Các hàm xử lý ---

def scrape_article_content(url: str) -> str:
    """
    Truy cập URL bài viết và trích xuất nội dung văn bản chính.
    LƯU Ý: Phần này có thể cần điều chỉnh vì cấu trúc HTML của mỗi trang web là khác nhau.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Báo lỗi nếu request không thành công

        soup = BeautifulSoup(response.content, 'html.parser')

        # Thử các selector khác nhau cho từng trang
        if 'vietstock.vn' in url:
            content_div = soup.find('div', id='content')
        elif 'cafef.vn' in url:
            content_div = soup.find('div', class_='content-detail')
        else:
            # Selector chung chung nếu không khớp
            content_div = soup.find('article') or soup.find('div', class_='content')

        if content_div:
            # Loại bỏ các thẻ không cần thiết như script, style
            for tag in content_div(['script', 'style', 'a', 'figure']):
                tag.decompose()
            return content_div.get_text(separator=' ', strip=True)
        return ""
    except requests.RequestException as e:
        logger.error(f"Lỗi khi truy cập url {url}: {e}")
        return ""

# --- Định nghĩa API Endpoint ---

@app.get(
    "/api/news-vectors",
    summary="Lấy tin tức và vector hóa nội dung",
    description="Lấy tối đa 5 bài viết mới nhất từ mỗi nguồn RSS, trích xuất toàn bộ nội dung, chia nhỏ và tạo vector cho từng đoạn."
)
def get_news_with_vectors():
    """
    Điểm cuối API chính để xử lý và trả về dữ liệu.
    """
    all_articles_data = []

    for feed_url in RSS_FEEDS:
        try:
            logger.info(f"Đang xử lý nguồn RSS: {feed_url}")
            feed = feedparser.parse(feed_url)

            # Giới hạn xử lý 5 bài viết đầu tiên trên mỗi nguồn để API phản hồi nhanh
            for entry in feed.entries[:5]:
                title = entry.get('title', 'Không có tiêu đề')
                link = entry.get('link')
                published = entry.get('published', 'Không có ngày xuất bản')
                source = feed.feed.get('title', 'Không rõ nguồn')

                if not link:
                    continue

                logger.info(f"Đang trích xuất nội dung từ: {link}")
                full_content = scrape_article_content(link)

                if not full_content:
                    logger.warning(f"Không lấy được nội dung cho bài viết: {title}")
                    continue

                # Chia nhỏ văn bản
                chunks = text_splitter.split_text(full_content)

                # Vector hóa từng đoạn
                # batch_size giúp xử lý hiệu quả hơn nếu có nhiều đoạn
                logger.info(f"Đang vector hóa {len(chunks)} đoạn văn bản...")
                vectors = model.encode(chunks).tolist() # Chuyển thành list để serialize JSON

                article_data = {
                    "source": source,
                    "title": title,
                    "link": link,
                    "published": published,
                    "content_chunks": [],
                }

                for i, chunk in enumerate(chunks):
                    article_data["content_chunks"].append({
                        "chunk_text": chunk,
                        "vector": vectors[i]
                    })

                all_articles_data.append(article_data)

        except Exception as e:
            logger.error(f"Lỗi khi xử lý nguồn {feed_url}: {e}")
            # Bỏ qua nguồn bị lỗi và tiếp tục với nguồn khác
            continue

    if not all_articles_data:
        raise HTTPException(status_code=404, detail="Không thể lấy được bài viết nào từ các nguồn RSS.")

    return {"articles": all_articles_data}

# Lệnh để chạy server local (dùng để kiểm thử):
# uvicorn main:app --reload
