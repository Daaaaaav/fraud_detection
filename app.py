import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

from preprocessing import preprocess_data
from randomforest import train_and_save_model, load_and_predict_bulk
from isolationforest import train_isolation_forest, detect_anomalies
from autoencoder_backend import train_autoencoder, predict_autoencoder, predict_autoencoder_manual
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')

FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

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
        logging.error(f'Error in /preprocess: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/train/randomforest', methods=['POST'])
def train_rf():
    try:
        data = request.get_json(force=True)
        model_name = data.get('name', 'rf_model')
        logging.info(f'Received data for Random Forest training: {data}')
        result = train_and_save_model(model_name)
        logging.info(f'Random Forest training complete: {result}')
        return jsonify(result)
    except Exception as e:
        logging.error(f'Error in /train/randomforest: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/train/isolationforest', methods=['POST'])
def train_iso():
    try:
        logging.info('Training Isolation Forest...')
        result = train_isolation_forest()
        logging.info(f'Isolation Forest training complete: {result}')
        return jsonify(result)
    except Exception as e:
        logging.error(f'Error in /train/isolationforest: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/train/autoencoder', methods=['POST'])
def train_autoencoder_route():
    try:
        logging.info("Training Autoencoder...")
        result = train_autoencoder()
        logging.info("Autoencoder training successful.")
        return jsonify({"message": "Autoencoder trained successfully", **result})
    except Exception as e:
        logging.error(f"Error training Autoencoder: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/randomforest/all', methods=['GET'])
def predict_rf_all():
    try:
        model_path = request.args.get('model', 'rf_model.pkl')
        logging.info(f"Loading Random Forest model from {model_path}")
        model = joblib.load(model_path)

        df = pd.read_csv('creditcard.csv')
        if 'Class' not in df.columns:
            raise ValueError("Missing 'Class' column in dataset.")

        X = df.drop(columns=['Class'])
        predictions = model.predict(X)

        result_df = X.copy()
        result_df['Prediction'] = predictions
        result_df['Prediction_Label'] = result_df['Prediction'].map({0: 'Not Fraudulent', 1: 'Fraudulent'})
        result_df['Actual'] = df['Class']
        result_df['Actual_Label'] = result_df['Actual'].map({0: 'Not Fraudulent', 1: 'Fraudulent'})

        fraud_count = sum(predictions)
        total = len(predictions)
        stats = {
            'total': total,
            'fraudulent': int(fraud_count),
            'non_fraudulent': int(total - fraud_count),
            'fraud_rate': round((fraud_count / total) * 100, 2)
        }

        logging.debug("Random Forest predictions done.")
        return jsonify({
            'predictions': result_df.head(100).to_dict(orient='records'),
            'stats': stats
        })
    except Exception as e:
        logging.exception("Error in /predict/randomforest/all")
        return jsonify({'error': str(e)}), 500
    
@app.route('/train/all_models', methods=['POST'])
def train_all_models():
    rf_metrics = train_and_save_model()
    iso_metrics = train_isolation_forest()
    auto_metrics = train_autoencoder()

    return jsonify({
        "rf": {
            "accuracy": rf_metrics["accuracy"],
            "precision": rf_metrics["precision"],
            "recall": rf_metrics["recall"],
            "f1_score": rf_metrics["f1_score"]
        },
        "iso": {
            "accuracy": iso_metrics["accuracy"],
            "precision": iso_metrics["precision"],
            "recall": iso_metrics["recall"],
            "f1_score": iso_metrics["f1_score"]
        },
        "auto": {
            "accuracy": auto_metrics["accuracy"],
            "precision": auto_metrics["precision"],
            "recall": auto_metrics["recall"],
            "f1_score": auto_metrics["f1_score"]
        }
    })


@app.route('/predict/isolationforest/all', methods=['GET'])
def predict_iso_all():
    try:
        logging.info("Predicting with Isolation Forest...")
        df = pd.read_csv('creditcard.csv')
        if 'Class' not in df.columns:
            raise ValueError("Missing 'Class' column in dataset.")

        X = df.drop(columns=['Class'])
        model = joblib.load('isolation_forest_model.pkl')
        predictions = model.predict(X)

        result_df = X.copy()
        result_df['Anomaly'] = predictions
        result_df['Anomaly_Label'] = result_df['Anomaly'].map({1: 'Normal', -1: 'Anomaly (Possible Fraud)'})
        result_df['Actual'] = df['Class']
        result_df['Actual_Label'] = result_df['Actual'].map({0: 'Not Fraudulent', 1: 'Fraudulent'})

        anomaly_count = sum(predictions == -1)
        total = len(predictions)
        stats = {
            'total': total,
            'anomalies_detected': int(anomaly_count),
            'normal': int(total - anomaly_count),
            'anomaly_rate': round((anomaly_count / total) * 100, 2)
        }

        logging.debug("Isolation Forest predictions complete.")
        return jsonify({
            'predictions': result_df.head(100).to_dict(orient='records'),
            'stats': stats
        })
    except Exception as e:
        logging.exception("Error in /predict/isolationforest/all")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/autoencoder/all', methods=['GET'])
def predict_autoencoder_route():
    try:
        results = predict_autoencoder()
        return jsonify(results)
    except Exception as e:
        logging.exception("Autoencoder prediction error")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/randomforest/manual', methods=['POST'])
def predict_rf_manual():
    try:
        logging.info("Received request for /predict/randomforest/manual")
        model = joblib.load('rf_model.pkl')
        user_data = request.get_json(force=True)
        logging.debug(f"User input: {user_data}")

        input_values = [user_data[feature] for feature in FEATURE_ORDER]
        df = pd.DataFrame([input_values], columns=FEATURE_ORDER)

        prediction = model.predict(df)[0]
        label = 'Fraudulent' if prediction == 1 else 'Not Fraudulent'

        return jsonify({'prediction': int(prediction), 'label': label})
    except Exception as e:
        logging.exception("Error in /predict/randomforest/manual")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/isolationforest/manual', methods=['POST'])
def predict_iso_manual():
    try:
        logging.info("Received request for /predict/isolationforest/manual")
        model = joblib.load('isolation_forest_model.pkl')
        user_data = request.get_json(force=True)
        logging.debug(f"User input: {user_data}")

        input_values = [user_data[feature] for feature in FEATURE_ORDER]
        df = pd.DataFrame([input_values], columns=FEATURE_ORDER)

        prediction = model.predict(df)[0]
        # Isolation Forest uses -1 for anomaly
        label = 'Fraudulent' if prediction == -1 else 'Not Fraudulent'

        return jsonify({'prediction': int(prediction), 'label': label})
    except Exception as e:
        logging.exception("Error in /predict/isolationforest/manual")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/autoencoder/manual', methods=['POST'])
def predict_autoencoder_manual_route():
    try:
        user_data = request.get_json(force=True)

        missing = [f for f in FEATURE_ORDER if f not in user_data]
        if missing:
            return jsonify({'error': f"Missing features: {missing}"}), 400

        try:
            input_values = [float(user_data[feature]) for feature in FEATURE_ORDER]
        except ValueError as ve:
            return jsonify({'error': f"Invalid numeric input: {str(ve)}"}), 400

        prediction_data = predict_autoencoder_manual(input_values)

        if "error" in prediction_data:
            return jsonify({"error": prediction_data["error"]}), 500

        prediction = prediction_data["is_fraud"]
        confidence = prediction_data.get("confidence", None) 

        label = 'Fraudulent' if prediction == 1 else 'Not Fraudulent'
        response = {'prediction': prediction, 'label': label}

        if confidence is not None:
            response['confidence'] = f"{confidence:.2%}"

        return jsonify(response)

    except Exception as e:
        logging.exception("Unhandled error in /predict/autoencoder/manual")
        return jsonify({'error': f"Internal server error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5006)
