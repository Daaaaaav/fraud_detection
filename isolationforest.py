import os
import json
import joblib
import numpy as np
import pandas as pd

from itertools import product
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from preprocessing import get_current_dataset, get_base_paths


def train_isolation_forest(dataset_filename=None, model_path='models/isolation_forest_model.pkl', threshold_path='models/threshold.json'):
    if dataset_filename is None:
        dataset_filename = get_current_dataset()
    paths = get_base_paths(dataset_filename)

    data = np.load(paths['processed'])
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    param_grid = {
        'n_estimators': [100],
        'max_samples': ['auto'],
        'contamination': [0.005],
        'max_features': [1.0]
    }

    best_model, best_f1, best_thresh, best_params = None, 0, None, None
    all_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

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

    # Save model and threshold
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)

    os.makedirs(os.path.dirname(threshold_path), exist_ok=True)
    with open(threshold_path, "w") as f:
        json.dump({"best_threshold": float(best_thresh)}, f)

    # Evaluate
    final_scores = -best_model.decision_function(X_test)
    final_preds = (final_scores > best_thresh).astype(int)

    return {
        'model': 'Isolation Forest',
        'best_params': best_params,
        'accuracy': float(accuracy_score(y_test, final_preds)),
        'precision': float(precision_score(y_test, final_preds, zero_division=0)),
        'recall': float(recall_score(y_test, final_preds, zero_division=0)),
        'f1_score': float(f1_score(y_test, final_preds, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, final_scores)),
        'best_threshold': float(best_thresh),
        'confusion_matrix': confusion_matrix(y_test, final_preds).tolist(),
        'message': f'Model saved to {model_path}'
    }


def detect_anomalies(dataset_filename=None, model_path='models/isolation_forest_model.pkl', threshold_path='models/threshold.json'):
    if dataset_filename is None:
        dataset_filename = get_current_dataset()
    if not os.path.exists(model_path):
        return {'error': f'Model file {model_path} not found.'}

    df = pd.read_csv(dataset_filename)
    paths = get_base_paths(dataset_filename)

    if not os.path.exists(paths["scaler"]):
        return {'error': f'Scaler file {paths["scaler"]} not found. Please run preprocessing.'}

    model = joblib.load(model_path)
    scaler = joblib.load(paths["scaler"])

    X = df.drop(columns=["Class"], errors='ignore')
    y_true = df["Class"] if "Class" in df.columns else None

    X_scaled = scaler.transform(X)
    scores = -model.decision_function(X_scaled)

    # Load threshold
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            best_thresh = json.load(f)["best_threshold"]
    else:
        best_thresh = np.percentile(scores, 99)

    preds = (scores > best_thresh).astype(int)
    df["predicted"] = preds
    df["is_fraud"] = preds == 1

    stats = {
        "model": "Isolation Forest",
        "fraud_count": int(df["is_fraud"].sum()),
    }

    if y_true is not None:
        stats.update({
            "accuracy": float(accuracy_score(y_true, preds)),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
            "f1_score": float(f1_score(y_true, preds, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, preds).tolist()
        })
    else:
        stats["warning"] = "No ground truth labels for evaluation."

    top_frauds = df[df["is_fraud"]].head(100)

    return {
        "top_frauds": top_frauds.to_dict(orient="records"),
        "stats": stats
    }
