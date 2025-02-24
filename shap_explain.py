import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ✅ Load trained model
model = joblib.load("models/optimized_stock_predictor.pkl")

# ✅ Load dataset
df = pd.read_csv("dataapi_output/AAPL_stock.csv")  # Ensure the correct file path
df = df[['Open', 'High', 'Low', 'Volume', 'Close']]  # Ensure correct column names

# ✅ Define input features
X = df[['Open', 'High', 'Low', 'Volume']]

# ✅ Create SHAP Explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X, check_additivity=False)  # ✅ Fixing additivity issue

# ✅ Save SHAP Summary Plot as PNG
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.savefig("static/shap_summary_plot.png", bbox_inches='tight')
print("✅ SHAP Summary plot saved as 'static/shap_summary_plot.png'")

# ✅ Save SHAP Force Plot as HTML
shap_html_path = "static/shap_force_plot.html"

# ✅ Extract Base Value Properly
base_value = explainer.expected_value
if isinstance(base_value, list) or isinstance(base_value, np.ndarray):  
    base_value = base_value[0]  # ✅ Fixing shape issues

# ✅ Convert SHAP Values to 2D for Force Plot
shap_values_array = np.array(shap_values)  
if len(shap_values_array.shape) == 1:  
    shap_values_array = shap_values_array.reshape(1, -1)

# ✅ Generate Force Plot
force_plot = shap.plots.force(
    base_value, shap_values_array[0, :], X.iloc[0, :], matplotlib=True
)

# ✅ Save Force Plot as HTML
shap.save_html(shap_html_path, force_plot)
print(f"✅ SHAP Force plot saved as '{shap_html_path}'")