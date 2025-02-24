import joblib
import pandas as pd

# Load Model
model = joblib.load("models/optimized_stock_predictor.pkl")

# Load Data
df = pd.read_csv("dataapi_output/QCOM_stock.csv")
FEATURES = ['Open', 'High', 'Low', 'Volume']

X = df[FEATURES]
latest_features = X.iloc[-1:].values.reshape(1, -1)  # Ensure correct shape

# Make Prediction
predicted_price = model.predict(latest_features)[0]
print(f"💰 **Predicted Closing Price: ${predicted_price:.2f}**")