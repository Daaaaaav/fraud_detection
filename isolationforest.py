import os
import json
import joblib
import numpy as np
import pandas as pd

from itertools import product
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss

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

    best_model, best_loss, best_thresh, best_params = None, float('inf'), None, None
    all_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

    for combo in all_combinations:
        params = dict(zip(param_names, combo))
        model = IsolationForest(**params, random_state=42)
        model.fit(X_train)
        scores = -model.decision_function(X_test)

        for t in np.linspace(scores.min(), scores.max(), 50):
            preds = (scores > t).astype(int)
            try:
                loss = log_loss(y_test, preds)
            except:
                loss = float('inf')
            if loss < best_loss:
                best_model, best_loss, best_thresh, best_params = model, loss, t, params

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)

    os.makedirs(os.path.dirname(threshold_path), exist_ok=True)
    with open(threshold_path, "w") as f:
        json.dump({"best_threshold": float(best_thresh)}, f)

    final_scores = -best_model.decision_function(X_test)
    final_preds = (final_scores > best_thresh).astype(int)

    return {
        'model': 'Isolation Forest',
        'best_params': best_params,
        'MSE': float(mean_squared_error(y_test, final_preds)),
        'MAE': float(mean_absolute_error(y_test, final_preds)),
        'CrossEntropyLoss': float(log_loss(y_test, final_preds)),
        'best_threshold': float(best_thresh),
        'message': f'Model saved to {model_path}'
    }
