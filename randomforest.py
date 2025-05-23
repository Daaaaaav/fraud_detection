import os
import numpy as np
import pandas as pd
import joblib
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from preprocessing import get_current_dataset, get_base_paths

def train_and_save_model(dataset_filename=None, model_path="models/random_forest.pkl"):
    if dataset_filename is None:
        dataset_filename = get_current_dataset()
    if not os.path.exists(dataset_filename):
        raise FileNotFoundError(f"Dataset not found: {dataset_filename}")

    paths = get_base_paths(dataset_filename)
    df = pd.read_csv(dataset_filename)

    if 'Class' not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    X = df.drop(columns=['Class']).select_dtypes(include=[np.number])
    y = df['Class']
    feature_names = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs(os.path.dirname(paths["scaler"]), exist_ok=True)
    joblib.dump(scaler, paths["scaler"])
    joblib.dump(feature_names, paths["features"])

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    os.makedirs(os.path.dirname(paths["processed"]), exist_ok=True)
    np.savez(paths["processed"], X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced']
    }

    best_model = None
    best_f1 = 0
    best_params = None

    for combo in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combo))
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds, zero_division=0)

        if f1 > best_f1:
            best_model = model
            best_f1 = f1
            best_params = params

    # Save full model bundle
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({
        "model": best_model,
        "scaler": scaler,
        "features": feature_names
    }, model_path)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    return {
        'model': 'Random Forest',
        'best_params': best_params,
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(best_f1),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'message': f'Model saved to {model_path}'
    }

def load_and_predict_bulk(dataset_filename=None, model_path="models/random_forest.pkl"):
    if dataset_filename is None:
        dataset_filename = get_current_dataset()
    if not os.path.exists(dataset_filename):
        return {'error': f'Dataset file not found: {dataset_filename}'}

    model_bundle = joblib.load(model_path)
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    features = model_bundle["features"]

    df = pd.read_csv(dataset_filename)
    if not all(f in df.columns for f in features):
        return {'error': f"Input data missing required features: {features}"}

    X = df[features].select_dtypes(include=[np.number])
    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    df["predicted"] = y_pred
    df["is_fraud"] = y_pred == 1

    stats = {
        "model": "Random Forest",
        "fraud_count": int(df["is_fraud"].sum()),
    }

    if "Class" in df.columns:
        y_true = df["Class"]
        stats.update({
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        })
    else:
        stats["warning"] = "No 'Class' column found. Cannot compute evaluation metrics."

    return {
        "top_frauds": df[df["is_fraud"]].head(100).to_dict(orient="records"),
        "stats": stats
    }
