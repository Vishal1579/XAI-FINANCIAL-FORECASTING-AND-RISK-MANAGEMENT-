import requests
import json
import os

# ✅ Replace with your actual GNews API key
GNEWS_API_KEY = "95201e4250171601f90cfcf95cb39af2"  

# ✅ Ask user for stock symbol
stock_symbol = input("📈 Enter the stock symbol (e.g., AAPL, TSLA, GOOGL): ").strip().upper()

# ✅ Ensure the output folder exists
os.makedirs("newsapi_output", exist_ok=True)

# ✅ GNews API URL
url = f"https://gnews.io/api/v4/search?q={stock_symbol}&lang=en&apikey={GNEWS_API_KEY}"

# ✅ Fetch News
response = requests.get(url)
data = response.json()

# ✅ Check if news articles exist
if "articles" in data:
    articles = data["articles"]

    # ✅ Print top 5 news articles
    print(f"\n📊 Latest News for {stock_symbol}:\n")
    for article in articles[:5]:
        print(f"📰 {article['title']}")
        print(f"📅 Date: {article['publishedAt']}")
        print(f"🔗 Link: {article['url']}\n")

    # ✅ Save news to a text file
    news_file = f"newsapi_output/{stock_symbol}_news.txt"
    with open(news_file, "w", encoding="utf-8") as f:
        for article in articles[:5]:
            f.write(f"📰 {article['title']}\n")
            f.write(f"📅 Date: {article['publishedAt']}\n")
            f.write(f"🔗 Link: {article['url']}\n\n")
    
    print(f"✅ News saved to '{news_file}'.")

else:
    print("❌ No news found or API limit reached!")