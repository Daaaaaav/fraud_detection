import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib
import os
from werkzeug.utils import secure_filename

from preprocessing import preprocess_data
from randomforest import train_and_save_model, load_and_predict_bulk
from isolationforest import train_isolation_forest, detect_anomalies
from autoencoder_backend import train_autoencoder, predict_autoencoder, predict_autoencoder_manual
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        result = preprocess_data(dataset_filename=file_path)
        return jsonify(result)
    except Exception as e:
        logging.exception('Error in /preprocess')
        return jsonify({'error': str(e)}), 500

@app.route('/train/randomforest', methods=['POST'])
def train_rf():
    try:
        data = request.get_json(force=True)
        model_name = data.get('name', 'rf_model')
        result = train_and_save_model(model_name)
        return jsonify(result)
    except Exception as e:
        logging.exception('Error in /train/randomforest')
        return jsonify({'error': str(e)}), 500

@app.route('/train/isolationforest', methods=['POST'])
def train_iso():
    try:
        result = train_isolation_forest()
        return jsonify(result)
    except Exception as e:
        logging.exception('Error in /train/isolationforest')
        return jsonify({'error': str(e)}), 500

@app.route('/train/autoencoder', methods=['POST'])
def train_autoencoder_route():
    try:
        result = train_autoencoder()
        return jsonify({"message": "Autoencoder trained successfully", **result})
    except Exception as e:
        logging.exception("Error training Autoencoder")
        return jsonify({"error": str(e)}), 500

@app.route('/train/all_models', methods=['POST'])
def train_all_models():
    try:
        rf_metrics = train_and_save_model()
        iso_metrics = train_isolation_forest()
        auto_metrics = train_autoencoder()

        return jsonify({
            "rf": rf_metrics,
            "iso": iso_metrics,
            "auto": auto_metrics
        })
    except Exception as e:
        logging.exception("Error in training all models")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/randomforest/all', methods=['GET'])
def predict_rf_all():
    try:
        model_path = request.args.get('model', 'rf_model.pkl')
        model = joblib.load(model_path)

        df = pd.read_csv('creditcard.csv')
        if 'Class' not in df.columns:
            raise ValueError("Missing 'Class' column.")

        X = df.drop(columns=['Class'])
        predictions = model.predict(X)

        result_df = X.copy()
        result_df['prediction'] = predictions
        result_df['label'] = result_df['prediction'].map({0: 'Not Fraudulent', 1: 'Fraudulent'})
        result_df['actual'] = df['Class']
        result_df['actual_label'] = result_df['actual'].map({0: 'Not Fraudulent', 1: 'Fraudulent'})

        stats = {
            'total': len(predictions),
            'fraudulent': int((predictions == 1).sum()),
            'non_fraudulent': int((predictions == 0).sum()),
            'fraud_rate': round((predictions == 1).sum() / len(predictions) * 100, 2)
        }

        return jsonify({
            'predictions': result_df.head(100).to_dict(orient='records'),
            'stats': stats
        })
    except Exception as e:
        logging.exception("Error in /predict/randomforest/all")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/isolationforest/all', methods=['GET'])
def predict_iso_all():
    try:
        df = pd.read_csv('creditcard.csv')
        if 'Class' not in df.columns:
            raise ValueError("Missing 'Class' column.")

        X = df.drop(columns=['Class'])
        model = joblib.load('isolation_forest_model.pkl')
        predictions = model.predict(X)

        result_df = X.copy()
        result_df['prediction'] = predictions
        result_df['label'] = result_df['prediction'].map({1: 'Not Fraudulent', -1: 'Fraudulent'})
        result_df['actual'] = df['Class']
        result_df['actual_label'] = result_df['actual'].map({0: 'Not Fraudulent', 1: 'Fraudulent'})

        stats = {
            'total': len(predictions),
            'fraudulent': int((predictions == -1).sum()),
            'non_fraudulent': int((predictions == 1).sum()),
            'fraud_rate': round((predictions == -1).sum() / len(predictions) * 100, 2)
        }

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
        model = joblib.load('rf_model.pkl')
        user_data = request.get_json(force=True)

        input_values = [user_data[feature] for feature in FEATURE_ORDER]
        df = pd.DataFrame([input_values], columns=FEATURE_ORDER)

        prediction = int(model.predict(df)[0])
        label = 'Fraudulent' if prediction == 1 else 'Not Fraudulent'

        return jsonify({'prediction': prediction, 'label': label})
    except Exception as e:
        logging.exception("Error in /predict/randomforest/manual")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/isolationforest/manual', methods=['POST'])
def predict_iso_manual():
    try:
        model = joblib.load('isolation_forest_model.pkl')
        user_data = request.get_json(force=True)

        input_values = [user_data[feature] for feature in FEATURE_ORDER]
        df = pd.DataFrame([input_values], columns=FEATURE_ORDER)

        prediction = int(model.predict(df)[0])
        label = 'Fraudulent' if prediction == -1 else 'Not Fraudulent'

        return jsonify({'prediction': prediction, 'label': label})
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

        input_values = [float(user_data[feature]) for feature in FEATURE_ORDER]
        prediction_data = predict_autoencoder_manual(input_values)

        if "error" in prediction_data:
            return jsonify({"error": prediction_data["error"]}), 500

        prediction = prediction_data["is_fraud"]
        confidence = prediction_data.get("confidence")

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