from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

# ✅ Step 1: Load Data
df = pd.read_csv("dataapi_output/AAPL_stock.csv")
df = df[['Open', 'High', 'Low', 'Volume', 'Close']]

# ✅ Step 2: Convert Regression into Classification (Up = 1, Down = -1, Neutral = 0)
df['Movement'] = df['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).shift(-1)
df = df.dropna()  # Drop NaN values

X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Movement']

# ✅ Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 4: Define Hyperparameter Grid
param_grid = {
    'n_estimators': [50, 100, 200],  
    'max_depth': [10, 20, None],  
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4]  
}

# ✅ Step 5: Perform Grid Search
print("🔄 Running Grid Search to Tune Hyperparameters...")
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# ✅ Step 6: Get Best Model
best_model = grid_search.best_estimator_
print(f"✅ Best Parameters: {grid_search.best_params_}")

# ✅ Step 7: Train & Evaluate the Optimized Model
train_acc = best_model.score(X_train, y_train)
test_acc = best_model.score(X_test, y_test)

print(f"📊 Training Accuracy: {train_acc:.2f}")
print(f"📊 Testing Accuracy: {test_acc:.2f}")

# ✅ Step 8: Save the Optimized Model
joblib.dump(best_model, "models/optimized_stock_predictor.pkl")
print("✅ Optimized Model Saved Successfully!")