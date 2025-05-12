import numpy as np 
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import joblib
import os
from itertools import product

df = pd.read_csv("creditcard.csv")
X = df.drop(columns=["Class"])
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

np.savez("processed_data.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

def train_isolation_forest():
    data = np.load('processed_data.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    param_grid = {
        'n_estimators': [100, 200],
        'max_samples': ['auto', 0.5, 0.8],
        'contamination': [0.0017, 0.005, 0.01],
        'max_features': [1.0, 0.8, 0.5]
    }

    all_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

    best_model = None
    best_f1 = 0
    best_params = None
    best_thresh = None

    for combo in all_combinations:
        params = dict(zip(param_names, combo))
        model = IsolationForest(**params, random_state=42)
        model.fit(X_train)

        anomaly_scores = -model.decision_function(X_test)

        for t in np.linspace(anomaly_scores.min(), anomaly_scores.max(), 50):
            preds = (anomaly_scores > t).astype(int)
            f1 = f1_score(y_test, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
                best_model = model
                best_params = params

    joblib.dump(best_model, 'isolation_forest_model.pkl')
    with open("threshold.json", "w") as f:
        json.dump({"best_threshold": float(best_thresh)}, f)

    # Evaluate final model
    anomaly_scores = -best_model.decision_function(X_test)
    y_pred_mapped = (anomaly_scores > best_thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred_mapped)

    stats = {
        'best_params': best_params,
        'total': int(len(y_pred_mapped)),
        'anomalies_detected': int(np.sum(y_pred_mapped)),
        'anomaly_rate': float(np.sum(y_pred_mapped) / len(y_pred_mapped)) * 100,
        'accuracy': float(accuracy_score(y_test, y_pred_mapped)),
        'precision': float(precision_score(y_test, y_pred_mapped, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_mapped, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred_mapped, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, anomaly_scores)),
        'best_threshold': float(best_thresh),
        'confusion_matrix': cm.tolist(),
        'true_class_distribution': {
            'fraud': int(np.sum(y_test)),
            'non_fraud': int(len(y_test) - np.sum(y_test))
        }
    }

    return stats


def detect_anomalies():
    df = pd.read_csv("creditcard.csv")

    model = joblib.load("isolation_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")

    X = df.drop(columns=["Class"])
    y_true = df["Class"]
    X_scaled = scaler.transform(X)
    anomaly_scores = -model.decision_function(X_scaled)

    if os.path.exists("threshold.json"):
        with open("threshold.json") as f:
            best_thresh = json.load(f)["best_threshold"]
    else:
        best_thresh = np.percentile(anomaly_scores, 99)  

    preds = (anomaly_scores > best_thresh).astype(int)
    df["predicted"] = preds
    df["is_fraud"] = df["predicted"] == 1

    fraudulent_df = df[df["is_fraud"]]
    non_fraudulent_df = df[~df["is_fraud"]]

    top_frauds = pd.concat([
        fraudulent_df.head(50),
        non_fraudulent_df.head(50)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    final_stats = {
        "model": "Isolation Forest (Tuned)",
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1_score": f1_score(y_true, preds, zero_division=0)
    }

    return {
        "top_frauds": top_frauds.to_dict(orient="records"),
        "stats": final_stats
    }
