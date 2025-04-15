import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# === Load Data ===
file_path = 'creditcard.csv'
df = pd.read_csv(file_path)

# === Data Overview ===
print("\n=== Data Overview ===")
print(df.describe())

# === Split Features and Labels ===
X = df.drop('Class', axis=1)
y = df['Class']

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n=== Data Shapes ===")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# === Scale Data ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Handle Class Imbalance with SMOTE ===
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print("\n=== After SMOTE Resampling ===")
print(f"X_train_resampled shape: {X_train_resampled.shape}")
print(f"y_train_resampled shape: {y_train_resampled.shape}")
print(f"Class distribution in y_train_resampled:\n{pd.Series(y_train_resampled).value_counts()}")

# === Compute Class Weights ===
class_weights = compute_class_weight(class_weight='balanced', 
                                     classes=np.array([0, 1]), 
                                     y=y_train_resampled)

class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print("\n=== Class Weights ===")
print(class_weight_dict)

# === Save Preprocessed Data ===
np.savez('processed_data.npz', 
         X_train=X_train_resampled, 
         X_test=X_test_scaled, 
         y_train=y_train_resampled, 
         y_test=y_test)

print("\nPreprocessing complete. Data saved to 'processed_data.npz'")
