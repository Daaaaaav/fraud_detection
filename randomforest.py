import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

def train_and_save_model(model_name='rf_model'):
    data = np.load('processed_data.npz')
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weights_dict = {0: class_weights[0], 1: class_weights[1]}

    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=weights_dict)
    clf.fit(X_train, y_train)
    joblib.dump(clf, f'{model_name}.pkl')

    y_pred = clf.predict(X_test)

    metrics = {
        'model': 'Random Forest',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'message': f'Model saved as {model_name}.pkl',
        'train_samples': int(len(y_train)),
        'test_samples': int(len(y_test)),
    }

    return metrics

def load_and_predict_bulk(model_path='rf_model.pkl'):
    model = joblib.load(model_path)
    scaler = joblib.load('scaler.pkl')
    df = pd.read_csv("creditcard.csv")

    X = df.drop(columns=["Class"])
    y_true = df["Class"]

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    df["predicted_class"] = y_pred
    df["is_fraud"] = df["predicted_class"] == 1

    top_frauds = df[df["is_fraud"] == True].head(100)

    return {
        "top_frauds": top_frauds.to_dict(orient="records"),
        "stats": {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0)
        }
    }
