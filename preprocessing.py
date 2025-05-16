import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.combine import SMOTETomek
import joblib

CURRENT_DATASET = None

def get_base_paths(dataset_filename):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    base_name = os.path.splitext(os.path.basename(dataset_filename))[0]
    return {
        "dir_path": dir_path,
        "base_name": base_name,
        "scaler": os.path.join(dir_path, f'scaler_{base_name}.pkl'),
        "features": os.path.join(dir_path, f'feature_names_{base_name}.npy'),
        "processed": os.path.join(dir_path, f'processed_{base_name}.npz')
    }

def preprocess_data(dataset_filename='creditcard.csv'):
    global CURRENT_DATASET
    paths = get_base_paths(dataset_filename)

    file_path = dataset_filename if os.path.isabs(dataset_filename) else os.path.join(paths["dir_path"], dataset_filename)
    if not os.path.exists(file_path):
        return {'error': f'Dataset "{dataset_filename}" not found at {file_path}.'}

    df = pd.read_csv(file_path)
    X, y = df.drop('Class', axis=1), df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, paths["scaler"])
    np.save(paths["features"], X.columns.to_numpy())

    smote_tomek = SMOTETomek(random_state=42)
    X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_scaled, y_train)

    np.savez(paths["processed"],
             X_train=X_train_resampled,
             X_test=X_test_scaled,
             y_train=y_train_resampled,
             y_test=y_test)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=y_train_resampled
    )

    sample_df = pd.DataFrame(X_train_resampled[:5], columns=X.columns)
    sample_df['Class'] = y_train_resampled[:5].values

    CURRENT_DATASET = dataset_filename

    return {
        'sample': sample_df.to_dict(orient='records'),
        'info': {
            'message': f'Preprocessing complete for {dataset_filename}.',
            'shapes': {
                'X_train': X_train_resampled.shape,
                'X_test': X_test_scaled.shape,
                'y_train': y_train_resampled.shape,
                'y_test': y_test.shape
            },
            'class_weights': {
                '0': float(class_weights[0]),
                '1': float(class_weights[1])
            }
        }
    }
