from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from preprocessing import preprocess_data
from randomforest import train_and_save_model, load_and_predict
from isolationforest import detect_anomalies, train_isolation_forest
from randomforest import train_xgboost

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return 'Fraud Detection API is running.'

@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        result = preprocess_data()
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

@app.route('/train/xgboost', methods=['POST'])
def train_xgb():
    try:
        result = train_xgboost()
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

@app.route('/predict/randomforest', methods=['POST'])
def predict_rf():
    try:
        data = request.get_json()
        prediction = load_and_predict(data)
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect/anomaly', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        result = detect_anomalies(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
