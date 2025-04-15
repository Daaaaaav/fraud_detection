from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from preprocessing import preprocess_data 
from randomforest import train_and_save_model, load_and_predict  
from isolationforest import detect_anomalies, train_isolation_forest
from xgboost import XGBClassifier  

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        result = preprocess_data()
        print(result) 
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train/randomforest', methods=['POST'])
def train_rf():
    try:
        data = request.get_json(force=True)
        name = data.get('name', 'random_forest_model')
        result = train_and_save_model(name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train/isolationforest', methods=['POST'])
def train_iso():
    try:
        result = train_isolation_forest()  
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/randomforest/all', methods=['GET'])
def predict_rf_all():
    try:
        df = pd.read_csv('creditcard.csv')  # atau pakai path absolut
        X = df.drop(columns=['Class'])
        model = joblib.load('random_forest_model.pkl')
        predictions = model.predict(X)

        # Gabungkan hasil prediksi dengan data asli (ambil sebagian kalau terlalu besar)
        result = X.copy()
        result['Prediction'] = predictions
        result['Actual'] = df['Class']

        return jsonify(result.head(100).to_dict(orient='records'))  # hanya 100 baris untuk efisiensi
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/isolationforest/all', methods=['GET'])
def predict_iso_all():
    try:
        df = pd.read_csv('creditcard.csv')
        X = df.drop(columns=['Class'])
        model = joblib.load('isolation_forest_model.pkl')
        predictions = model.predict(X)

        result = X.copy()
        result['Anomaly'] = predictions
        result['Actual'] = df['Class']

        return jsonify(result.head(100).to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/xgboost', methods=['POST'])
def predict_xgb():
    try:
        data = request.get_json()
        model = joblib.load('xgboost_model.pkl')  
        X = pd.DataFrame(data)  
        y_pred = model.predict(X) 
        return jsonify({'prediction': int(y_pred[0])})  
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect/anomaly', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        df = pd.DataFrame(data) 
        result = detect_anomalies(df) 
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
