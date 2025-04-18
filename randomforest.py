import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_save_model(model_name='rf_model'):
    # Load preprocessed data
    data = np.load('processed_data.npz')
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, f'{model_name}.pkl')

    # Predict
    y_pred = clf.predict(X_test)

    # Metrics
    metrics = {
        'model': 'Random Forest',
        'accuracy': accuracy_score(y_test, y_pred) * 100,  # as a percentage
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'message': f'Model saved as {model_name}.pkl',
        'train_samples': int(len(y_train)),
        'test_samples': int(len(y_test)),
    }

    return metrics


def load_and_predict(input_data, model_path='rf_model.pkl'):
    # Load model
    model = joblib.load(model_path)

    # Convert input dict to array (single sample)
    X = np.array([list(input_data.values())])
    prediction = model.predict(X)

    return int(prediction[0])