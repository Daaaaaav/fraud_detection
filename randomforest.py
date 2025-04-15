import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load the preprocessed data
def load_data():
    file_path = 'preprocessed_data.csv'  # Make sure this matches the output file from preprocessing.py
    data = pd.read_csv(file_path)
    print("\n=== Loaded Data Sample ===")
    print(data.head())
    
    # Ensure the column name matches the target column from preprocessing.py
    print("\n=== Data Columns ===")
    print(data.columns)
    
    # Fix case sensitivity in column names
    if 'class' in data.columns:
        data.rename(columns={'class': 'Class'}, inplace=True)
    
    # Split features and target
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    return X, y

def train_random_forest(X, y):
    # Split data into training and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"\nTraining set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # Initialize and train the RandomForest model
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        class_weight='balanced'  # Handles class imbalance
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Model Evaluation
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Accuracy Score ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Save the model for future use
    joblib.dump(model, 'random_forest_model.pkl')
    print("\nModel saved as 'random_forest_model.pkl'")

if __name__ == "__main__":
    try:
        X, y = load_data()
        train_random_forest(X, y)
    except Exception as e:
        print(f"\nError: {e}")
