import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
import joblib
import json

def train_isolation_forest():
    print("Loading data...")
    df = pd.read_csv('creditcard.csv')

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTEENN
    print("\nBalancing data using SMOTEENN...")
    smote_enn = SMOTEENN(random_state=42)
    X_train_bal, y_train_bal = smote_enn.fit_resample(X_train, y_train)

    # Train XGBoost
    print("Training XGBoost Classifier...")
    xgb = XGBClassifier(
        scale_pos_weight=len(y_train_bal[y_train_bal == 0]) / len(y_train_bal[y_train_bal == 1]),
        max_depth=4,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train_bal, y_train_bal, eval_set=[(X_test, y_test)], verbose=False)

    y_scores_xgb = xgb.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores_xgb)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_threshold = thresholds[np.argmax(f1_scores)]
    y_pred_xgb = (y_scores_xgb >= best_threshold).astype(int)

    # Train Isolation Forest
    print("\nTraining Isolation Forest (unsupervised)...")
    iso_forest = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination='auto',
        max_features=1.0,
        bootstrap=False,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train)
    y_pred_if = iso_forest.predict(X_test)
    y_pred_if = np.where(y_pred_if == -1, 1, 0)

    # Save models
    joblib.dump(xgb, 'xgboost_model.pkl')
    joblib.dump(iso_forest, 'isolation_forest_model.pkl')

    # Save predictions
    pd.DataFrame({
        'y_true': y_test.values,
        'xgb_pred': y_pred_xgb,
        'xgb_score': y_scores_xgb,
        'isoforest_pred': y_pred_if
    }).to_csv('model_predictions.csv', index=False)

    # Save XGBoost evaluation
    xgb_eval = {
        'confusion_matrix': confusion_matrix(y_test, y_pred_xgb).tolist(),
        'classification_report': classification_report(y_test, y_pred_xgb, digits=4, output_dict=True),
        'roc_auc': roc_auc_score(y_test, y_scores_xgb),
        'best_threshold': float(best_threshold)
    }
    with open('xgboost_evaluation.json', 'w') as f:
        json.dump(xgb_eval, f, indent=4)

    # Save Isolation Forest evaluation
    try:
        iso_auc = roc_auc_score(y_test, -iso_forest.decision_function(X_test))
    except:
        iso_auc = None

    isoforest_eval = {
        'confusion_matrix': confusion_matrix(y_test, y_pred_if).tolist(),
        'classification_report': classification_report(y_test, y_pred_if, digits=4, output_dict=True),
        'roc_auc': iso_auc
    }
    with open('isolation_forest_evaluation.json', 'w') as f:
        json.dump(isoforest_eval, f, indent=4)

    # Save threshold tuning data
    pd.DataFrame({
        'threshold': thresholds,
        'precision': precision[:-1],
        'recall': recall[:-1],
        'f1': f1_scores[:-1]
    }).to_csv('xgb_thresholds_analysis.csv', index=False)

    print("\n All results saved successfully.")

    return {
        'xgb_model': 'xgboost_model.pkl',
        'iso_forest_model': 'isolation_forest_model.pkl',
        'message': 'Both models trained and saved successfully.'
    }


if __name__ == '__main__':
    train_isolation_forest()


def detect_anomalies(data):
    # Assuming `data` is a DataFrame with the same features as the training data
    iso_forest = joblib.load('isolation_forest_model.pkl')  # Load the trained Isolation Forest model
    predictions = iso_forest.predict(data)  # Predict anomalies
    # Anomalies are labeled as -1, normal points as 1, so we invert the labels
    anomalies = np.where(predictions == -1, 1, 0)
    
    return {
        'anomalies': anomalies.tolist()
    }
