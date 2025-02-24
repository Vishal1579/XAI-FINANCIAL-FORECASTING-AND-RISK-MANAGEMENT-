import requests
import pandas as pd

# Define API Key and Stock Symbol
API_KEY = "8BU6J0WYYD6VU9XF"  # Your Alpha Vantage API Key
symbol = "AMD"  # Change this to any stock symbol

# Free Alpha Vantage API URL (only daily prices)
url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}&outputsize=full"

# Fetch Data
response = requests.get(url)
data = response.json()

# Convert to DataFrame
if "Time Series (Daily)" in data:
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
    df = df.astype(float)

    # Rename Columns
    df = df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. volume": "Volume"
    })

    # Convert index to datetime
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()


    # Save Data to CSV
    df.to_csv(f"dataapi_output/{symbol}_stock.csv", index=False)

    print(f"✅ {symbol} full historical stock data saved successfully!")
else:
    print(f"❌ Error fetching data: {data}")