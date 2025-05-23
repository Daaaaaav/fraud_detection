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
    base_name = os.path.splitext(os.path.basename(dataset_filename))[0]
    return {
        "processed": f"data/processed/{base_name}_processed.npz",
        "scaler": f"models/{base_name}_scaler.pkl",
        "features": f"models/{base_name}_features.pkl"
    }

def get_current_dataset():
    return CURRENT_DATASET

def preprocess_data(dataset_filename=None):
    global CURRENT_DATASET
    paths = get_base_paths(dataset_filename)

    # Construct the full path to the raw dataset
    file_path = dataset_filename if os.path.isabs(dataset_filename) else os.path.join("data/raw", dataset_filename)

    if not os.path.exists(file_path):
        return {'error': f'Dataset "{dataset_filename}" not found at {file_path}.'}

    # Create necessary directories if they don't exist
    os.makedirs(os.path.dirname(paths["processed"]), exist_ok=True)
    os.makedirs(os.path.dirname(paths["scaler"]), exist_ok=True)
    os.makedirs("models", exist_ok=True)  # In case it's not covered

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return {'error': f'Failed to read CSV: {str(e)}'}

    if 'Class' not in df.columns:
        return {'error': '"Class" column is required in the dataset.'}

    try:
        X = df.drop('Class', axis=1)
        y = df['Class']
    except Exception as e:
        return {'error': f'Error during splitting features and labels: {str(e)}'}

    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    except Exception as e:
        return {'error': f'Error during train/test split: {str(e)}'}

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler and features
    joblib.dump(scaler, paths["scaler"])
    np.save(paths["features"], X.columns.to_numpy())

    # Resample using SMOTETomek
    try:
        smote_tomek = SMOTETomek(random_state=42)
        X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_scaled, y_train)
    except Exception as e:
        return {'error': f'Error during resampling: {str(e)}'}
    
    # Save processed data
    try:
        np.savez(paths["processed"],
                 X_train=X_train_resampled,
                 X_test=X_test_scaled,
                 y_train=y_train_resampled,
                 y_test=y_test)
    except Exception as e:
        return {'error': f'Failed to save processed data: {str(e)}'}

    # Compute class weights
    try:
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_resampled),
            y=y_train_resampled
        )
        class_weight_dict = {str(cls): float(w) for cls, w in zip(np.unique(y_train_resampled), class_weights)}
    except Exception as e:
        class_weight_dict = {}
        print(f"Warning: Failed to compute class weights: {e}")

    # Sample preview
    try:
        sample_df = pd.DataFrame(X_train_resampled[:5], columns=X.columns)
    except Exception:
        sample_df = pd.DataFrame(X_train_resampled[:5])
    sample_df['Class'] = y_train_resampled[:5].values

    CURRENT_DATASET = dataset_filename

    return {
        'sample': sample_df.to_dict(orient='records'),
        'info': {
            'message': f'Preprocessing complete for "{dataset_filename}".',
            'shapes': {
                'X_train': X_train_resampled.shape,
                'X_test': X_test_scaled.shape,
                'y_train': y_train_resampled.shape,
                'y_test': y_test.shape
            },
            'class_weights': class_weight_dict
        }
    }

def load_preprocessed(dataset_filename):
    paths = get_base_paths(dataset_filename)

    if not os.path.exists(paths["processed"]):
        return {'error': f'Processed file not found for {dataset_filename}. Run preprocess_data() first.'}

    try:
        data = np.load(paths["processed"])
        X_train = data['X_train']
        y_train = data['y_train']
        return X_train, y_train
    except Exception as e:
        return {'error': f'Failed to load processed dataset: {str(e)}'}
