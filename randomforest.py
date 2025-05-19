import os
import numpy as np
import pandas as pd
import joblib
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from preprocessing import get_current_dataset, get_base_paths, preprocess_data  # Adjust based on your project structure

def train_and_save_model(dataset_filename=None, model_path="models/random_forest.pkl"):
    if dataset_filename is None:
        dataset_filename = get_current_dataset()
    paths = get_base_paths(dataset_filename)

    # Load dataset manually (assuming it's a CSV)
    df = pd.read_csv(dataset_filename)

    if 'Class' not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column as the target.")

    X = df.drop(columns=['Class'])
    y = df['Class']

    # Ensure only numeric features are used
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Expected X to be a pandas DataFrame.")
    X = X.select_dtypes(include=[np.number])

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler
    os.makedirs(os.path.dirname(paths["scaler"]), exist_ok=True)
    joblib.dump(scaler, paths["scaler"])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    # Save preprocessed data
    os.makedirs(os.path.dirname(paths["processed"]), exist_ok=True)
    np.savez(paths["processed"], X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced']
    }

    best_model = None
    best_f1 = 0
    best_params = None

    all_combos = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

    for combo in all_combos:
        params = dict(zip(param_names, combo))
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds, zero_division=0)

        if f1 > best_f1:
            best_model = model
            best_f1 = f1
            best_params = params

    # Save the best model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    return {
        'model': 'Random Forest',
        'best_params': best_params,
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'message': f'Model saved to {model_path}'
    }


def load_and_predict_bulk(dataset_filename=None, model_path="models/random_forest.pkl"):
    if dataset_filename is None:
        dataset_filename = get_current_dataset()

    if not os.path.exists(model_path):
        return {'error': f'Model file {model_path} not found.'}

    paths = get_base_paths(dataset_filename)
    if not os.path.exists(paths["scaler"]):
        return {'error': f'Scaler file {paths["scaler"]} not found. Please run preprocessing.'}

    model = joblib.load(model_path)
    scaler = joblib.load(paths["scaler"])
    df = pd.read_csv(dataset_filename)

    y_true = df["Class"] if "Class" in df.columns else None
    X = df.drop(columns=["Class"], errors='ignore')
    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    df["predicted"] = y_pred
    df["is_fraud"] = df["predicted"] == 1

    stats = {
        "model": "Random Forest",
        "fraud_count": int(df["is_fraud"].sum()),
    }

    if y_true is not None:
        stats.update({
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        })
    else:
        stats["warning"] = "No ground truth labels for evaluation."

    top_frauds = df[df["is_fraud"]].head(100)

    return {
        "top_frauds": top_frauds.to_dict(orient="records"),
        "stats": stats
    }
