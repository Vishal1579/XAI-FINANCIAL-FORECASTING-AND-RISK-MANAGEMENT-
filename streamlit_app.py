import streamlit as st
import pandas as pd
import joblib
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import os
import subprocess  # To call Ollama

# Set Page Config
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# Load Model
MODEL_PATH = "models/optimized_stock_predictor.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    st.sidebar.success(f"✅ Model loaded: `{type(model).__name__}`")
else:
    st.sidebar.error("❌ Model not found! Train it first.")

# Sidebar Dropdown for Stock Selection
stock_options = ["AAPL", "GOOGL", "IBM", "INTC", "MSFT", "NVDA", "QCOM", "TSLA", "WMT","AMD"]
selected_stock = st.sidebar.selectbox("📈 Select a Stock:", stock_options, index=stock_options.index("QCOM"))
st.sidebar.subheader(f"📈 Selected Stock: {selected_stock}")

# Load Stock Data
stock_path = f"dataapi_output/{selected_stock}_stock.csv"

if os.path.exists(stock_path):
    df = pd.read_csv(stock_path)

    if df.empty:
        st.warning(f"No data found for {selected_stock}!")
    else:
        st.write(f"### {selected_stock} Stock Data")
        st.dataframe(df.tail())

        # Define Features & Target
        FEATURES = ['Open', 'High', 'Low', 'Volume']
        TARGET = 'Close'
        
        X = df[FEATURES]
        y = df[TARGET]

        # **Predicted Price Logic**
        latest_features = X.iloc[-1:].values.reshape(1, -1)  # Ensure correct shape
        predicted_price = model.predict(latest_features)[0]
        st.success(f"💰 **Predicted Closing Price: ${predicted_price:.2f}**")

        # **SHAP Explanation**
        st.subheader("📊 SHAP Explanation")
        try:
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)

            plt.figure(figsize=(8, 6))
            shap.summary_plot(shap_values, X, show=False)
            shap_plot_path = "static/shap_summary_plot.png"
            plt.savefig(shap_plot_path, bbox_inches="tight")
            st.image(shap_plot_path, caption="SHAP Feature Importance")
        except Exception as e:
            st.error(f"SHAP Error: {e}")

        # **LIME Explanation**
        st.subheader("🔍 LIME Explanation")
        try:
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X.values, mode="regression", feature_names=FEATURES
            )
            lime_exp = lime_explainer.explain_instance(X.iloc[-1].values, model.predict, num_features=4)
            lime_html_path = "static/lime_explanation.html"
            lime_exp.save_to_file(lime_html_path)
            st.components.v1.html(open(lime_html_path).read(), height=400)
        except Exception as e:
            st.error(f"LIME Error: {e}")

        # **Load Stock-Specific Explanation**
        llm_explain_path = f"llm_output/{selected_stock}_explanation.txt"
        if os.path.exists(llm_explain_path):
            with open(llm_explain_path, "r") as f:
                explanation_content = f.read()
            st.subheader(f"🤖 LLM Analysis ({selected_stock})")
            st.text(explanation_content)
        else:
            st.warning("LLM explanation not found!")

        # **News Sentiment (if available)**
        news_path = f"newsapi_output/{selected_stock}_news.txt"
        if os.path.exists(news_path):
            with open(news_path, "r") as f:
                news_content = f.read()
            st.subheader("📰 News Sentiment")
            st.write(news_content)
        else:
            st.warning(f"No news found for {selected_stock}.")

        # **🔮 LLM Prediction & Explanation (via Ollama)**
        st.subheader("📡 LLM-Enhanced Stock Analysis")
        try:
            llama_prompt = f"Analyze the stock trend for {selected_stock} with a predicted closing price of ${predicted_price:.2f}."
            
            # Use Ollama (Llama 3.2)
            result = subprocess.run(["ollama", "run", "llama3", llama_prompt], capture_output=True, text=True)
            if result.returncode == 0:
                st.write(result.stdout)
            else:
                st.error("❌ Error running Llama 3.2 in Ollama!")
        except Exception as e:
            st.error(f"LLM Error: {e}")

else:
    st.error(f"❌ Stock file `{stock_path}` not found!")