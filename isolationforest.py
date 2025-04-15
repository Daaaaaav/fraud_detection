import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve
)
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
import joblib
import json

# Load data
print("Loading data...")
df = pd.read_csv('creditcard.csv')

# Split features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- XGBOOST PIPELINE (with SMOTEENN) ---
print("\nBalancing data using SMOTEENN...")
smote_enn = SMOTEENN(random_state=42)
X_train_bal, y_train_bal = smote_enn.fit_resample(X_train, y_train)

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

print("Predicting probabilities...")
y_scores_xgb = xgb.predict_proba(X_test)[:, 1]

# Threshold tuning
precision, recall, thresholds = precision_recall_curve(y_test, y_scores_xgb)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_threshold = thresholds[np.argmax(f1_scores)]
y_pred_xgb = (y_scores_xgb >= best_threshold).astype(int)

# --- ISOLATION FOREST PIPELINE ---
print("\nTraining Isolation Forest (unsupervised)...")
iso_forest = IsolationForest(
    n_estimators=100,
    max_samples='auto',
    contamination='auto',  # Optional: or float(sum(y_train)/len(y_train))
    max_features=1.0,
    bootstrap=False,
    random_state=42,
    n_jobs=-1
)

iso_forest.fit(X_train)
y_pred_if = iso_forest.predict(X_test)
y_pred_if = np.where(y_pred_if == -1, 1, 0)

# --- EVALUATION SECTION ---
print("\n--- XGBOOST EVALUATION ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb, digits=4))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_scores_xgb):.4f}")

print("\n--- ISOLATION FOREST EVALUATION ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_if))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_if, digits=4))
try:
    print(f"ROC AUC Score (Isolation Forest): {roc_auc_score(y_test, -iso_forest.decision_function(X_test)):.4f}")
except:
    print("ROC AUC Score: N/A")

# --- SAVE RESULTS SECTION ---

# Save models
joblib.dump(xgb, 'xgboost_model.pkl')
joblib.dump(iso_forest, 'isolation_forest_model.pkl')

# Save predictions
predictions_df = pd.DataFrame({
    'y_true': y_test.values,
    'xgb_pred': y_pred_xgb,
    'xgb_score': y_scores_xgb,
    'isoforest_pred': y_pred_if
})
predictions_df.to_csv('model_predictions.csv', index=False)

# Save XGBoost evaluation metrics
xgb_eval = {
    'confusion_matrix': confusion_matrix(y_test, y_pred_xgb).tolist(),
    'classification_report': classification_report(y_test, y_pred_xgb, digits=4, output_dict=True),
    'roc_auc': roc_auc_score(y_test, y_scores_xgb),
    'best_threshold': float(best_threshold)
}
with open('xgboost_evaluation.json', 'w') as f:
    json.dump(xgb_eval, f, indent=4)

# Save Isolation Forest evaluation metrics
isoforest_eval = {
    'confusion_matrix': confusion_matrix(y_test, y_pred_if).tolist(),
    'classification_report': classification_report(y_test, y_pred_if, digits=4, output_dict=True),
    'roc_auc': roc_auc_score(y_test, -iso_forest.decision_function(X_test))
}
with open('isolation_forest_evaluation.json', 'w') as f:
    json.dump(isoforest_eval, f, indent=4)

# Save threshold tuning results (optional)
thresholds_df = pd.DataFrame({
    'threshold': thresholds,
    'precision': precision[:-1],
    'recall': recall[:-1],
    'f1': f1_scores[:-1]
})
thresholds_df.to_csv('xgb_thresholds_analysis.csv', index=False)

print("\n✔ All results saved successfully.")
