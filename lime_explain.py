import lime
import lime.lime_tabular
import joblib
import pandas as pd
import numpy as np

# ✅ Load Optimized Model
model = joblib.load("models/optimized_stock_predictor.pkl")

# ✅ Load Dataset
df = pd.read_csv("dataapi_output/AAPL_stock.csv")
df = df[['Open', 'High', 'Low', 'Volume', 'Close']]

X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

# ✅ Initialize LIME Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X.values, feature_names=X.columns, class_names=['Close'], mode='regression'
)

# ✅ Pick a Random Data Point
idx = np.random.randint(0, X.shape[0])
sample = X.iloc[idx].values.reshape(1, -1)

# ✅ Explain Prediction
exp = explainer.explain_instance(sample[0], model.predict)
exp.save_to_file("static/lime_explanation.html")

print(f"✅ LIME Explanation saved as 'static/lime_explanation.html' for index {idx}")