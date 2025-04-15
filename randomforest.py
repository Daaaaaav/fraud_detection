import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_and_save_model(name='random_forest_model'):
    data = np.load('processed_data.npz')
    X_train, y_train = data['X_train'], data['y_train']

    model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    filename = f'{name}.pkl'
    joblib.dump(model, filename)

    return {'message': f'Model trained and saved as {filename}'}

def load_and_predict(input_data, filename='random_forest_model.pkl'):
    model = joblib.load(filename)
    prediction = model.predict([list(input_data.values())])
    return prediction[0]
