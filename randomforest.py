import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from preprocessing import preprocess_data, get_base_paths

def train_and_save_model(dataset_filename='creditcard.csv', model_name='rf_model'):
    preprocess_data(dataset_filename)
    paths = get_base_paths(dataset_filename)
    data = np.load(paths["processed"])

    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weights_dict = {i: w for i, w in enumerate(class_weights)}

    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=weights_dict)
    clf.fit(X_train, y_train)
    joblib.dump(clf, f'{model_name}.pkl')

    y_pred = clf.predict(X_test)

    return {
        'model': 'Random Forest',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'message': f'Model saved as {model_name}.pkl',
    }

def load_and_predict_bulk(model_path='rf_model.pkl', dataset_filename='creditcard.csv'):
    paths = get_base_paths(dataset_filename)
    model = joblib.load(model_path)
    scaler = joblib.load(paths["scaler"])
    df = pd.read_csv(dataset_filename)

    X = df.drop(columns=["Class"])
    y_true = df["Class"]
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    df["predicted_class"] = y_pred
    df["is_fraud"] = df["predicted_class"] == 1
    top_frauds = df[df["is_fraud"]].head(100)

    return {
        "top_frauds": top_frauds.to_dict(orient="records"),
        "stats": {
            "model": "Random Forest",
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0)
        }
    }
