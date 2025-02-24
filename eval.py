import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, precision_score, recall_score, f1_score
)

# ✅ Load Optimized Model
model = joblib.load("models/optimized_stock_predictor.pkl")
print(f"✅ Loaded model type: {type(model)}")

# ✅ Load Dataset
df = pd.read_csv("dataapi_output/AAPL_stock.csv")
df = df[['Open', 'High', 'Low', 'Volume', 'Close']]  # Ensure correct columns

# ✅ Define Input and Target
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

# ✅ Predictions
predictions = model.predict(X)

# ✅ Calculate Errors
mae = mean_absolute_error(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))
r2 = r2_score(y, predictions)

# ✅ Cross-Validation
cv_mae = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error'))
cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2')

print(f"📊 Testing MAE: {mae:.2f}")
print(f"📊 Testing RMSE: {rmse:.2f}")
print(f"📊 Testing R² Score: {r2:.3f}")
print(f"📊 Cross-Validation MAE: {cv_mae.mean():.2f}")
print(f"📊 Cross-Validation RMSE: {cv_rmse.mean():.2f}")
print(f"📊 Cross-Validation R²: {cv_r2.mean():.3f}")

# ✅ Feature Importance Plot
importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=features, palette="viridis")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.savefig("static/feature_importance.png")
print("✅ Feature Importance plot saved as 'static/feature_importance.png'")

# ✅ Prediction vs. Actual Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y, y=predictions, alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Prediction vs Actual Stock Prices")
plt.savefig("static/pred_vs_actual.png")
print("✅ Prediction vs Actual plot saved as 'static/pred_vs_actual.png'")

# ✅ Confusion Matrix (Convert Regression to Binary Classes)
threshold = np.median(y)  # Use median as threshold
y_binary = (y > threshold).astype(int)
pred_binary = (predictions > threshold).astype(int)
conf_matrix = confusion_matrix(y_binary, pred_binary)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Binarized Regression)")
plt.savefig("static/confusion_matrix.png")
print("✅ Confusion Matrix saved as 'static/confusion_matrix.png'")

# ✅ Precision, Recall, and F1-Score
precision = precision_score(y_binary, pred_binary)
recall = recall_score(y_binary, pred_binary)
f1 = f1_score(y_binary, pred_binary)

print(f"📊 Precision: {precision:.2f}")
print(f"📊 Recall: {recall:.2f}")
print(f"📊 F1 Score: {f1:.2f}")

# ✅ Show Results
plt.show()