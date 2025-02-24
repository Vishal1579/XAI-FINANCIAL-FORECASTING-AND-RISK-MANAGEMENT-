import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 🔹 Your GNews API Key
GNEWS_API_KEY = "YOUR_GNEWS_API_KEY"

# 🔹 Fetch News for a Given Stock Symbol
def fetch_news(stock_symbol):
    url = f"https://gnews.io/api/v4/search?q={stock_symbol}&token={GNEWS_API_KEY}&lang=en&max=10"
    response = requests.get(url)
    data = response.json()
    
    if "articles" not in data:
        print("⚠️ No news found. Check API key & quota.")
        return []
    
    articles = [
        {"title": article["title"], "description": article["description"], "url": article["url"]}
        for article in data["articles"]
    ]
    
    return articles

# 🔹 Perform Sentiment Analysis on News
def analyze_sentiment(news_articles):
    analyzer = SentimentIntensityAnalyzer()
    
    for article in news_articles:
        text = article["title"] + ". " + article["description"]
        sentiment_score = analyzer.polarity_scores(text)["compound"]
        article["sentiment_score"] = sentiment_score
        article["sentiment"] = (
            "Positive" if sentiment_score > 0.05 else
            "Negative" if sentiment_score < -0.05 else
            "Neutral"
        )
    
    return news_articles

# 🔹 Save News & Sentiment Data to CSV
def save_to_csv(news_articles, stock_symbol):
    df = pd.DataFrame(news_articles)
    df.to_csv(f"dataapi_output/{stock_symbol}_news_sentiment.csv", index=False)
    print(f"✅ News sentiment data saved to 'dataapi_output/{stock_symbol}_news_sentiment.csv'")

# 🔹 Main Function
if __name__ == "__main__":
    stock_symbol = input("📈 Enter stock symbol (e.g., AAPL, TSLA, GOOGL): ").upper()
    news_articles = fetch_news(stock_symbol)
    
    if news_articles:
        news_articles = analyze_sentiment(news_articles)
        save_to_csv(news_articles, stock_symbol)