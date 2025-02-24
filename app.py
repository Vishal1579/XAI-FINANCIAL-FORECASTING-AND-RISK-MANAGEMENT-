import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ✅ Ensure model exists
MODEL_PATH = "models/optimized_stock_predictor.pkl"
if not os.path.exists(MODEL_PATH):
    print("❌ ERROR: Model file not found. Run model_training.py first!")
    exit()

# ✅ Load the trained model
model = joblib.load(MODEL_PATH)
print(f"✅ Model loaded: {type(model)}")

# ✅ Ensure dataset exists
DATA_PATH = "dataapi_output/AAPL_stock.csv"
if not os.path.exists(DATA_PATH):
    print("❌ ERROR: Dataset file not found. Check path:", DATA_PATH)
    exit()

# ✅ Load dataset for feature reference
df = pd.read_csv(DATA_PATH)
FEATURES = ['Open', 'High', 'Low', 'Volume']  # Must match model input features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.content_type == 'application/json':  # ✅ Handle JSON request
            data = request.get_json()
            print("🔹 Received JSON input:", data)
        else:  # ✅ Handle Form submission (HTML form data)
            data = {
                "Open": float(request.form.get('open')),
                "High": float(request.form.get('high')),
                "Low": float(request.form.get('low')),
                "Volume": float(request.form.get('volume'))
            }
            print("🔹 Received Form input:", data)

        # ✅ Ensure all features are present
        missing_features = [f for f in FEATURES if f not in data]
        if missing_features:
            return jsonify({'error': f'Missing features: {missing_features}'})

        # ✅ Convert input to DataFrame
        input_data = pd.DataFrame([data])
        print("🔹 Converted to DataFrame:", input_data)

        # ✅ Make prediction
        prediction = model.predict(input_data)[0]
        print("✅ Prediction Output:", prediction)

        return jsonify({'predicted_price': prediction})
    except Exception as e:
        print("❌ ERROR in Prediction:", str(e))
        return jsonify({'error': str(e)})

@app.route('/results', methods=['POST'])
def results():
    try:
        stock_symbol = request.form['stock_symbol']
        open_price = float(request.form['open'])
        high_price = float(request.form['high'])
        low_price = float(request.form['low'])
        volume = float(request.form['volume'])

        # ✅ Convert input into DataFrame
        input_data = pd.DataFrame([[open_price, high_price, low_price, volume]], columns=FEATURES)
        prediction = model.predict(input_data)[0]

        return render_template('results.html', stock_symbol=stock_symbol, predicted_price=prediction)
    except Exception as e:
        print("❌ ERROR in Results Page:", str(e))
        return render_template('results.html', stock_symbol=None, predicted_price=None, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)