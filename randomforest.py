import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_save_model(model_name='random_forest_model'):
    data = np.load('processed_data.npz')
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, f'{model_name}.pkl')

    y_pred = clf.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'message': f'{model_name}.pkl saved',
        'train_samples': int(len(y_train)),
        'test_samples': int(len(y_test)),
    }

    return metrics


def load_and_predict(input_data):
    model = joblib.load('random_forest_model.pkl')
    X = np.array([list(input_data.values())])
    prediction = model.predict(X)
    return int(prediction[0])
