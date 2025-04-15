import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

def preprocess_data():
    try:
        # === Load Data ===
        file_path = 'creditcard.csv'
        df = pd.read_csv(file_path)

        # === Split Features and Labels ===
        X = df.drop('Class', axis=1)
        y = df['Class']

        # === Train-Test Split ===
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # === Scale Data ===
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # === Handle Class Imbalance with SMOTE ===
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

        # === Compute Class Weights ===
        class_weights = compute_class_weight(class_weight='balanced',
                                             classes=np.array([0, 1]),
                                             y=y_train_resampled)
        class_weight_dict = {0: float(class_weights[0]), 1: float(class_weights[1])}

        # === Save Preprocessed Data ===
        np.savez('processed_data.npz', 
                 X_train=X_train_resampled, 
                 X_test=X_test_scaled, 
                 y_train=y_train_resampled, 
                 y_test=y_test)

        # === Return Status Info ===
        return {
            'message': 'Preprocessing complete.',
            'shapes': {
                'X_train': X_train_resampled.shape,
                'X_test': X_test_scaled.shape,
                'y_train': y_train_resampled.shape,
                'y_test': y_test.shape
            },
            'class_weights': class_weight_dict,
            'resampled_class_distribution': {
                '0': int((y_train_resampled == 0).sum()),
                '1': int((y_train_resampled == 1).sum())
            }
        }
    
    except Exception as e:
        return {'error': str(e)}
