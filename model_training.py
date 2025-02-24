from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import joblib
import numpy as np

# Load Data
df = pd.read_csv("dataapi_output/AAPL_stock.csv")
df = df[['Open', 'High', 'Low', 'Volume', 'Close']]

# Shift Close to predict next day's value (fix data leakage)
df['Target'] = df['Close'].shift(-1)  # Predict next day's Close
df = df.dropna()  # Remove last row with NaN target

# Introduce more noise
noise = np.random.normal(0, 0.2, df[['Open', 'High', 'Low', 'Volume']].shape)  # Noise level 0.2
df[['Open', 'High', 'Low', 'Volume']] += noise

# Define Input and Target
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Target']

# Split Data (60-40 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train Model (strong regularization)
model = RandomForestRegressor(
    n_estimators=10,  # Few trees
    max_depth=3,  # Shallow trees
    min_samples_split=20,  # High split threshold
    min_samples_leaf=15,  # Large leaves
    max_features='log2',  # Fewer features per split
    bootstrap=True,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Calculate Metrics
train_mae = mean_absolute_error(y_train, train_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)
test_r2 = r2_score(y_test, test_predictions)

print(f"📊 Training MAE: {train_mae:.2f}")
print(f"📊 Testing MAE: {test_mae:.2f}")
print(f"📊 Testing R²: {test_r2:.3f}")

# Pseudo-Accuracy (optional)
mean_y_test = y_test.mean()
pseudo_accuracy = 1 - (test_mae / mean_y_test)
print(f"📈 Pseudo-Accuracy (1 - MAE/mean): {pseudo_accuracy:.2%}")

# Save Model
joblib.dump(model, "models/optimized_stock_predictor.pkl")
print("✅ Optimized model saved successfully!")