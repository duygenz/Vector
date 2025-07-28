import uvicorn
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import feedparser
import asyncio

# List of your RSS feeds
RSS_FEEDS = [
    "https://vietstock.vn/830/chung-khoan/co-phieu.rss",
    "https://cafef.vn/thi-truong-chung-khoan.rss",
    "https://vietstock.vn/145/chung-khoan/y-kien-chuyen-gia.rss",
    "https://vietstock.vn/737/doanh-nghiep/hoat-dong-kinh-doanh.rss",
    "https://vietstock.vn/1328/dong-duong/thi-truong-chung-khoan.rss"
]

# Initialize the FastAPI app
app = FastAPI()

# Load a lightweight model for creating vectors.
# 'all-MiniLM-L6-v2' is a good balance of size and performance.
model = SentenceTransformer('all-MiniLM-L6-v2')

def fetch_and_parse_feeds():
    """Fetches news items from all RSS feeds."""
    all_entries = []
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        all_entries.extend(feed.entries)
    return all_entries

@app.get("/news-vectors")
async def get_news_vectors():
    """
    API endpoint to get news articles as vectors.
    """
    # Fetch the latest news
    articles = fetch_and_parse_feeds()

    # Prepare data for vectorization (e.g., using titles)
    titles = [entry.title for entry in articles]

    # Generate vectors
    vectors = model.encode(titles)

    # Combine titles with their vectors
    response_data = [
        {"title": title, "vector": vector.tolist()}
        for title, vector in zip(titles, vectors)
    ]

    return {"news": response_data}

if __name__ == "__main__":
    # This part is for local testing, not for Render
    uvicorn.run(app, host="0.0.0.0", port=8000)

