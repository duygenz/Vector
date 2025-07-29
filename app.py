import asyncio
import requests
import feedparser
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from typing import List, Dict, Any

# ----------------------------------
# KHỞI TẠO ỨNG DỤNG VÀ MÔ HÌNH
# ----------------------------------

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="News API with Vectorization",
    description="API để lấy tin tức từ các nguồn RSS và tạo vector cho nội dung.",
    version="1.0.0"
)

# Cấu hình CORS (Cross-Origin Resource Sharing)
# Cho phép tất cả các nguồn gốc, phương thức và tiêu đề.
# Trong môi trường production, bạn nên giới hạn lại các nguồn gốc được phép.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Danh sách các nguồn RSS
RSS_FEEDS = [
    "https://cafef.vn/thi-truong-chung-khoan.rss",
    "https://vneconomy.vn/chung-khoan.rss",
    "https://vneconomy.vn/tai-chinh.rss",
    "https://vneconomy.vn/thi-truong.rss",
    "https://vneconomy.vn/nhip-cau-doanh-nghiep.rss",
    "https://vneconomy.vn/tin-moi.rss",
    "https://cafebiz.vn/rss/cau-chuyen-kinh-doanh.rss"
]

# Tải mô hình sentence transformer nhỏ gọn.
# 'all-MiniLM-L6-v2' là một mô hình rất hiệu quả và có kích thước chỉ khoảng 90MB.
try:
    print("Đang tải mô hình vector hóa...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Tải mô hình thành công!")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    model = None

# ----------------------------------
# CÁC HÀM HỖ TRỢ
# ----------------------------------

def get_full_content(url: str) -> str:
    """
    Hàm lấy toàn bộ nội dung văn bản từ một URL bài báo.
    Sử dụng BeautifulSoup để phân tích cú pháp HTML và trích xuất văn bản.
    """
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Loại bỏ các thẻ không mong muốn
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        
        # Lấy văn bản từ body
        text = soup.body.get_text(separator=' ', strip=True)
        return text
    except requests.RequestException as e:
        print(f"Lỗi khi truy cập URL {url}: {e}")
        return ""

async def parse_feed(feed_url: str) -> List[Dict[str, Any]]:
    """
    Phân tích một nguồn RSS và trả về danh sách các bài báo đã được xử lý.
    """
    news_items = []
    parsed_feed = feedparser.parse(feed_url)
    for entry in parsed_feed.entries:
        full_content = await asyncio.to_thread(get_full_content, entry.link)
        news_items.append({
            "title": entry.title,
            "link": entry.link,
            "summary": entry.summary,
            "published": entry.get("published", "N/A"),
            "source": parsed_feed.feed.get("title", "N/A"),
            "full_content": full_content
        })
    return news_items

# ----------------------------------
# ĐỊNH NGHĨA CÁC ENDPOINT API
# ----------------------------------

@app.get("/", summary="Endpoint kiểm tra trạng thái", tags=["Status"])
async def read_root():
    """Endpoint gốc để kiểm tra xem API có hoạt động không."""
    return {"status": "ok", "message": "Chào mừng bạn đến với API Tin tức!"}

@app.get("/news", summary="Lấy tin tức hàng loạt", tags=["News"])
async def get_all_news():
    """
    Lấy tin tức từ tất cả các nguồn RSS đã định cấu hình.
    Endpoint này không tạo vector để trả về nhanh chóng.
    """
    try:
        tasks = [parse_feed(url) for url in RSS_FEEDS]
        results = await asyncio.gather(*tasks)
        # Làm phẳng danh sách các kết quả
        all_news = [item for sublist in results for item in sublist]
        return {"count": len(all_news), "news": all_news}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/news/vectors", summary="Lấy tin tức hàng loạt kèm vector", tags=["News with Vectors"])
async def get_news_with_vectors():
    """
    Lấy tin tức từ tất cả các nguồn RSS và tạo vector "full-context" cho mỗi bài báo.
    Quá trình này có thể mất một chút thời gian tùy thuộc vào số lượng bài báo.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Mô hình vector hóa không khả dụng.")
        
    try:
        # Lấy dữ liệu tin tức
        print("Bắt đầu lấy dữ liệu tin tức...")
        tasks = [parse_feed(url) for url in RSS_FEEDS]
        results = await asyncio.gather(*tasks)
        all_news = [item for sublist in results for item in sublist]
        print(f"Đã lấy được {len(all_news)} bài báo.")

        # Chuẩn bị danh sách nội dung để tạo vector
        contents_to_vectorize = [item['full_content'] for item in all_news if item['full_content']]
        
        if not contents_to_vectorize:
            return {"count": 0, "news": []}

        # Tạo vector hàng loạt
        print("Bắt đầu tạo vector hàng loạt...")
        vectors = model.encode(contents_to_vectorize, show_progress_bar=True)
        print("Tạo vector thành công!")

        # Gán vector vào lại các bài báo tương ứng
        vector_index = 0
        for item in all_news:
            if item['full_content']:
                item['vector'] = vectors[vector_index].tolist() # Chuyển numpy array thành list để trả về JSON
                vector_index += 1
            else:
                item['vector'] = None

        return {"count": len(all_news), "news": all_news}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Đã xảy ra lỗi: {str(e)}")

# ----------------------------------
# CHẠY ỨNG DỤNG (KHI CHẠY LOCAL)
# ----------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

