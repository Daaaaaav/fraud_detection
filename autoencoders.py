import keras_tuner as kt
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, f1_score
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from preprocessing import load_and_preprocess

# Load preprocessed data
file_path = 'xgb_thresholds_analysis.csv'
X_train_sm, y_train_sm, X_test, y_test = load_and_preprocess(file_path)

# ✅ Train autoencoder ONLY on class 0 (legitimate transactions)
X_train_legit = X_train_sm[y_train_sm == 0]
print("✅ Legit samples for training:", X_train_legit.shape)

# Define model builder for Keras Tuner
def build_autoencoder(hp):
    input_dim = X_train_legit.shape[1]
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoder = Dense(hp.Int('encoder_units_1', min_value=16, max_value=128, step=16), activation="relu")(input_layer)
    encoder = Dropout(0.2)(encoder)
    encoder = Dense(hp.Int('encoder_units_2', min_value=16, max_value=64, step=16), activation="relu")(encoder)
    
    # Decoder
    decoder = Dense(hp.Int('decoder_units_1', min_value=16, max_value=64, step=16), activation="relu")(encoder)
    output_layer = Dense(input_dim, activation="sigmoid")(decoder)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    
    autoencoder.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', [0.001, 0.0005])),
        loss=MeanSquaredError()
    )
    
    return autoencoder

# Set up the tuner
tuner = kt.Hyperband(
    build_autoencoder,
    objective='val_loss',
    max_epochs=20,
    factor=3,
    directory='autoencoder_tuning',
    project_name='creditcard_fraud',
    overwrite=True  # ensures a clean search
)

# Perform the hyperparameter search
tuner.search(
    X_train_legit, X_train_legit,
    epochs=10,
    validation_split=0.2,
    batch_size=32
)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters found:", best_hps.values)

# Load previous model if exists
model_path = 'best_autoencoder.h5'
if os.path.exists(model_path):
    best_model = load_model(model_path)
    print("✅ Loaded existing model from", model_path)
else:
    best_model = tuner.hypermodel.build(best_hps)
    print("🆕 Created a new model with best hyperparameters")

# Setup EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Fit the model and capture history
history = best_model.fit(
    X_train_legit, X_train_legit,
    epochs=50,
    validation_split=0.2,
    batch_size=32,
    callbacks=[early_stopping]
)

# Save the trained model
best_model.save(model_path)
print("💾 Saved trained model to", model_path)

# === Evaluate on Test Set ===
reconstructions = best_model.predict(X_test)
loss = np.mean(np.square(X_test - reconstructions), axis=1)

# Calculate threshold based on train reconstruction errors
train_reconstructions = best_model.predict(X_train_legit)
train_loss = np.mean(np.square(X_train_legit - train_reconstructions), axis=1)
threshold = np.percentile(train_loss, 95)
print(f"📉 Threshold for anomaly detection (95th percentile): {threshold:.6f}")

# Save the threshold
with open("best_threshold.txt", "w") as f:
    f.write(str(threshold))

# Predict anomalies
y_pred = (loss > threshold).astype(int)

# === Evaluation Metrics ===
print("🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n🧾 Classification Report:")
print(classification_report(y_test, y_pred))

# Save reconstruction losses and predictions
results_df = pd.DataFrame({
    'loss': loss,
    'true_label': y_test,
    'predicted_label': y_pred
})
results_df.to_csv('autoencoder_loss_analysis.csv', index=False)
print("📁 Saved reconstruction loss and predictions to 'autoencoder_loss_analysis.csv'")

# === Plot Training History ===
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Autoencoder Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('training_loss.png')
plt.show()

# === Plot 1: Reconstruction Error Distribution ===
plt.figure(figsize=(10, 6))
plt.hist(loss[y_test == 0], bins=50, alpha=0.6, label='Legit (0)', color='skyblue')
plt.hist(loss[y_test == 1], bins=50, alpha=0.6, label='Fraud (1)', color='salmon')
plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold = {threshold:.4f}')
plt.title('Reconstruction Error Distribution')
plt.xlabel('Reconstruction Error (Loss)')
plt.ylabel('Number of Samples')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('threshold_distribution.png')
plt.show()

# === Plot 2: Precision-Recall vs Threshold ===
precisions, recalls, thresholds_pr = precision_recall_curve(y_test, loss)
plt.figure(figsize=(10, 6))
plt.plot(thresholds_pr, precisions[:-1], label='Precision', color='blue')
plt.plot(thresholds_pr, recalls[:-1], label='Recall', color='green')
plt.axvline(threshold, color='red', linestyle='--', label=f'Chosen Threshold = {threshold:.4f}')
plt.title('Precision & Recall vs Threshold')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('precision_recall_vs_threshold.png')
plt.show()

# === Plot 3: F1 Score vs Threshold ===
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1])
plt.figure(figsize=(10, 6))
plt.plot(thresholds_pr, f1_scores, color='purple', label='F1 Score')
plt.axvline(threshold, color='red', linestyle='--', label=f'Chosen Threshold = {threshold:.4f}')
plt.title('F1 Score vs Threshold')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('f1_vs_threshold.png')
plt.show()

# Report best F1 threshold
best_f1_idx = np.argmax(f1_scores)
best_f1_threshold = thresholds_pr[best_f1_idx]
print(f"🔥 Best F1 Threshold: {best_f1_threshold:.4f} with F1 Score: {f1_scores[best_f1_idx]:.4f}")
