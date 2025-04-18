import logging
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

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        logging.info('Received request for /preprocess')
        result = preprocess_data()
        logging.info('Preprocessing completed successfully')
        logging.debug(f'Preprocessing result: {result}')
        return jsonify(result)
    except Exception as e:
        logging.error(f'Error occurred in /preprocess: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/train/randomforest', methods=['POST'])
def train_rf():
    try:
        data = request.get_json(force=True)
        logging.info(f'Received data for /train/randomforest: {data}')
        name = data.get('name', 'rf_model')
        result = train_and_save_model(name)
        logging.info(f'Random Forest model training completed: {result}')
        return jsonify(result)
    except Exception as e:
        logging.error(f'Error occurred in /train/randomforest: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/train/isolationforest', methods=['POST'])
def train_iso():
    try:
        logging.info('Received request for /train/isolationforest')
        result = train_isolation_forest()
        logging.info(f'Isolation Forest model training completed: {result}')
        return jsonify(result)
    except Exception as e:
        logging.error(f'Error occurred in /train/isolationforest: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/predict/randomforest/all', methods=['GET'])
def predict_rf_all():
    try:
        model_path = request.args.get('model', 'rf_model.pkl')
        logging.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)

        logging.info("Loading dataset")
        df = pd.read_csv('creditcard.csv')
        logging.debug(f"Dataset columns: {df.columns.tolist()}")

        if 'Class' not in df.columns:
            raise ValueError("Column 'Class' not found in the dataset.")

        X = df.drop(columns=['Class'])
        predictions = model.predict(X)

        result = X.copy()
        result['Prediction'] = predictions
        result['Actual'] = df['Class']

        return jsonify(result.head(100).to_dict(orient='records'))
    except Exception as e:
        logging.exception("Error during /predict/randomforest/all")
        return jsonify({'error': str(e)}), 500


@app.route('/predict/isolationforest/all', methods=['GET'])
def predict_iso_all():
    try:
        logging.info('Received request for /predict/isolationforest/all')
        df = pd.read_csv('creditcard.csv')
        X = df.drop(columns=['Class'])
        model = joblib.load('isolation_forest_model.pkl')
        predictions = model.predict(X)

        result = X.copy()
        result['Anomaly'] = predictions
        result['Actual'] = df['Class']

        logging.debug(f'Isolation Forest predictions: {result.head(5)}')
        return jsonify(result.head(100).to_dict(orient='records'))
    except Exception as e:
        logging.error(f'Error occurred in /predict/isolationforest/all: {e}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
