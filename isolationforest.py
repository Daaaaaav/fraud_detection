import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def train_isolation_forest():
    data = np.load('processed_data.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    y_test = data['y_test']

    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X_train)
    joblib.dump(model, 'isolation_forest_model.pkl')

    y_pred = model.predict(X_test)
    y_test_mapped = np.where(y_test == 1, -1, 1)

    metrics = {
        'model': 'Isolation Forest',
        'accuracy': accuracy_score(y_test_mapped, y_pred),
        'precision': precision_score(y_test_mapped, y_pred, pos_label=-1, zero_division=0),
        'recall': recall_score(y_test_mapped, y_pred, pos_label=-1, zero_division=0),
        'f1_score': f1_score(y_test_mapped, y_pred, pos_label=-1, zero_division=0),
        'message': 'Isolation Forest trained and saved'
    }

    return metrics

def detect_anomalies():
    df = pd.read_csv("creditcard.csv")
    model = joblib.load("isolation_forest_model.pkl")
    scaler = joblib.load('scaler.pkl')

    X = df.drop(columns=["Class"])
    y_true = df["Class"]

    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    df["predicted"] = np.where(preds == -1, 1, 0)
    df["is_fraud"] = df["predicted"] == 1

    return {
        "top_frauds": df[df["is_fraud"]].head(100).to_dict(orient="records"),
        "stats": {
            "accuracy": accuracy_score(y_true, df["predicted"]),
            "precision": precision_score(y_true, df["predicted"], zero_division=0),
            "recall": recall_score(y_true, df["predicted"], zero_division=0),
            "f1_score": f1_score(y_true, df["predicted"], zero_division=0)
        }
    }
