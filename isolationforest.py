import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

def train_isolation_forest():
    data = np.load('processed_data.npz')
    X_train = data['X_train']

    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X_train)
    joblib.dump(model, 'isolation_forest_model.pkl')
    return { 'message': 'Isolation Forest trained and saved' }


def detect_anomalies(df):
    model = joblib.load('isolation_forest_model.pkl')
    X = df.values
    predictions = model.predict(X)
    return { 'anomalies': predictions.tolist() }