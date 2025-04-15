import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

def train_and_save_model(model_name='random_forest_model'):
    data = np.load('processed_data.npz')
    X_train = data['X_train']
    y_train = data['y_train']

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, f'{model_name}.pkl')
    
    return { 'message': f'{model_name}.pkl saved', 'train_samples': int(len(y_train)) }


def load_and_predict(input_data):
    model = joblib.load('random_forest_model.pkl')
    X = np.array([list(input_data.values())])
    prediction = model.predict(X)
    return int(prediction[0])