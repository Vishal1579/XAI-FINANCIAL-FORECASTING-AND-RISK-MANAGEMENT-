import pandas as pd
import ollama
import json

# Stock Symbol (Change this to test different stocks)
STOCK_SYMBOL = "AMD"

# Load stock data
stock_data = pd.read_csv(f"dataapi_output/{STOCK_SYMBOL}_stock.csv")

# Load news data
with open(f"newsapi_output/{STOCK_SYMBOL}_news.txt", "r") as file:
    news_text = file.read()

# Load SHAP explanation (if exists)
try:
    with open(f"shap_explanation/{STOCK_SYMBOL}_shap_summary.txt", "r") as file:
        shap_text = file.read()
except FileNotFoundError:
    shap_text = "SHAP analysis not found."

# Load LIME explanation (if exists)
try:
    with open(f"lime_explanation/{STOCK_SYMBOL}_lime_summary.txt", "r") as file:
        lime_text = file.read()
except FileNotFoundError:
    lime_text = "LIME explanation not found."

# Generate input for LLM
prompt = f"""
Analyze the stock movements for {STOCK_SYMBOL} over the last few days based on:
1. **Stock Price Data**: {stock_data.tail(5).to_string()}
2. **Recent News Headlines**: {news_text}
3. **SHAP Explanation**: {shap_text}
4. **LIME Explanation**: {lime_text}

Summarize the reasons for the stock's movement, and explain whether it's likely to go up or down based on these factors.
"""

# Send to Ollama (Llama 3) or GPT API
response = ollama.chat(
    model="llama3",
    messages=[{"role": "user", "content": prompt}],
)

# Save output
output_text = response['message']['content']
output_path = f"llm_output/{STOCK_SYMBOL}_explanation.txt"

with open(output_path, "w") as output_file:
    output_file.write(output_text)

# Print results
print("\n📊 **LLM Explanation:**\n")
print(output_text)
print(f"\n✅ Explanation saved as: {output_path}")