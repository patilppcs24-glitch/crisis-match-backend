import os
from dotenv import load_dotenv
from firecrawl import FirecrawlApp

load_dotenv()

api_key = os.getenv("FIRECRAWL_API_KEY")

if not api_key:
    raise ValueError("FIRECRAWL_API_KEY not found")

app = FirecrawlApp(api_key=api_key)

def load_firecrawl_data():
    urls = [
        "https://www.redcross.org/get-help/how-to-prepare-for-emergencies/types-of-emergencies.html"
    ]

    documents = []

    for url in urls:
        try:
            data = app.scrape_url(url)
            content = data.get("content", "")
            if content:
                documents.append(content)
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    return documents