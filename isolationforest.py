import numpy as np
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
        'accuracy': accuracy_score(y_test_mapped, y_pred),
        'precision': precision_score(y_test_mapped, y_pred, pos_label=-1, zero_division=0),
        'recall': recall_score(y_test_mapped, y_pred, pos_label=-1, zero_division=0),
        'f1_score': f1_score(y_test_mapped, y_pred, pos_label=-1, zero_division=0),
        'message': 'Isolation Forest trained and saved'
    }

    return metrics


def detect_anomalies(df):
    model = joblib.load('isolation_forest_model.pkl')
    X = df.values
    predictions = model.predict(X)
    return { 'anomalies': predictions.tolist() }
