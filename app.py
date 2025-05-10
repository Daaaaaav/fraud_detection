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
from autoencoder_backend import train_autoencoder, predict_autoencoder
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
    
@app.route("/metrics/randomforest", methods=["GET"])
def metrics_random_forest():
    try:
        result = load_and_predict_bulk()
        return jsonify(result["stats"])
    except Exception as e:
        logging.exception("Error in /metrics/randomforest")
        return jsonify({"error": str(e)}), 500


@app.route("/metrics/isolationforest", methods=["GET"])
def metrics_isolation_forest():
    try:
        result = detect_anomalies()
        return jsonify(result["stats"])
    except Exception as e:
        logging.exception("Error in /metrics/isolationforest")
        return jsonify({"error": str(e)}), 500


@app.route("/metrics/autoencoder", methods=["GET"])
def metrics_autoencoder():
    try:
        df = pd.read_csv("creditcard.csv")
        X = df.drop(columns=["Class"])
        y_true = df["Class"]

        model = tf.keras.models.load_model("models/autoencoder.h5", compile=False)
        scaler = joblib.load("models/scaler.pkl")
        with open("models/threshold.txt", "r") as f:
            threshold = float(f.read())

        X_scaled = scaler.transform(X)
        X_pred = model.predict(X_scaled, verbose=0)
        mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
        y_pred = mse > threshold

        stats = {
            "model": "Autoencoder",
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0)
        }

        return jsonify(stats)
    except Exception as e:
        logging.exception("Error in /metrics/autoencoder")
        return jsonify({"error": str(e)}), 500


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
        logging.info("Received request for /predict/autoencoder/manual")
        user_data = request.get_json(force=True)
        logging.debug(f"User input: {user_data}")

        input_values = [user_data[feature] for feature in FEATURE_ORDER]
        prediction_data = predict_autoencoder_manual(input_values)

        if "error" in prediction_data:
            return jsonify({"error": prediction_data["error"]}), 500

        prediction = int(prediction_data["is_fraud"])
        label = 'Fraudulent' if prediction == 1 else 'Not Fraudulent'

        return jsonify({'prediction': prediction, 'label': label})
    except Exception as e:
        logging.exception("Error in /predict/autoencoder/manual")
        return jsonify({'error': str(e)}), 500

def predict_autoencoder_manual(user_input):
    try:
        logging.info("Predicting using manual Autoencoder input")
        model = tf.keras.models.load_model("models/autoencoder.h5", compile=False)
        scaler = joblib.load("models/scaler.pkl")

        X_manual = np.array(user_input).reshape(1, -1)
        X_scaled = scaler.transform(X_manual)
        X_pred = model.predict(X_scaled, verbose=0)

        mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)[0]
        with open("models/threshold.txt", "r") as f:
            threshold = float(f.read())

        is_fraud = mse > threshold
        return {
            "is_fraud": bool(is_fraud)
        }
    except Exception as e:
        logging.exception("Error in predict_autoencoder_manual")
        return {"error": str(e)}

if __name__ == '__main__':
    app.run(debug=True, port=5006)
