import numpy as np
import pandas as pd
import json
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from itertools import product
from preprocessing import preprocess_data, get_base_paths

def train_isolation_forest(dataset_filename='creditcard.csv'):
    preprocess_data(dataset_filename)
    paths = get_base_paths(dataset_filename)
    data = np.load(paths["processed"])

    X_train, X_test = data['X_train'], data['X_test']
    y_test = data['y_test']

    param_grid = {
        'n_estimators': [100],
        'max_samples': ['auto'],
        'contamination': [0.005],
        'max_features': [1.0]
    }

    all_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

    best_model, best_f1, best_params, best_thresh = None, 0, None, None

    for combo in all_combinations:
        params = dict(zip(param_names, combo))
        model = IsolationForest(**params, random_state=42)
        model.fit(X_train)
        scores = -model.decision_function(X_test)

        for t in np.linspace(scores.min(), scores.max(), 50):
            preds = (scores > t).astype(int)
            f1 = f1_score(y_test, preds, zero_division=0)
            if f1 > best_f1:
                best_model, best_f1, best_thresh, best_params = model, f1, t, params

    joblib.dump(best_model, 'isolation_forest_model.pkl')
    with open("threshold.json", "w") as f:
        json.dump({"best_threshold": float(best_thresh)}, f)

    final_scores = -best_model.decision_function(X_test)
    final_preds = (final_scores > best_thresh).astype(int)

    return {
        'best_params': best_params,
        'accuracy': float(accuracy_score(y_test, final_preds)),
        'precision': float(precision_score(y_test, final_preds, zero_division=0)),
        'recall': float(recall_score(y_test, final_preds, zero_division=0)),
        'f1_score': float(f1_score(y_test, final_preds, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, final_scores)),
        'best_threshold': float(best_thresh),
        'confusion_matrix': confusion_matrix(y_test, final_preds).tolist()
    }

def detect_anomalies(dataset_filename='creditcard.csv'):
    df = pd.read_csv(dataset_filename)
    paths = get_base_paths(dataset_filename)

    model = joblib.load("isolation_forest_model.pkl")
    scaler = joblib.load(paths["scaler"])

    X = df.drop(columns=["Class"])
    y_true = df["Class"]
    X_scaled = scaler.transform(X)
    scores = -model.decision_function(X_scaled)

    if os.path.exists("threshold.json"):
        with open("threshold.json") as f:
            best_thresh = json.load(f)["best_threshold"]
    else:
        best_thresh = np.percentile(scores, 99)

    preds = (scores > best_thresh).astype(int)
    df["predicted"] = preds
    df["is_fraud"] = preds == 1

    return {
        "frauds": df[df["is_fraud"]].to_dict(orient="records"),
        "stats": {
            "total": len(df),
            "fraud_count": int(df["is_fraud"].sum()),
            "accuracy": accuracy_score(y_true, preds),
            "precision": precision_score(y_true, preds, zero_division=0),
            "recall": recall_score(y_true, preds, zero_division=0),
            "f1_score": f1_score(y_true, preds, zero_division=0)
        }
    }
