# autoencoder.py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

MODEL_PATH = "best_autoencoder.h5"
THRESHOLD_PATH = "best_threshold.txt"

def load_data():
    df = pd.read_csv("creditcard.csv")
    X = df.drop(columns=["Class"])
    y = df["Class"].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def train_autoencoder_model():
    X_scaled, y, _ = load_data()
    X_normal = X_scaled[y == 0]
    X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)
    input_dim = X_train.shape[1]

    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(64, activation="relu", activity_regularizer=regularizers.l2(1e-5))(input_layer)
    encoded = layers.Dropout(0.2)(encoded)
    encoded = layers.Dense(32, activation="relu")(encoded)
    encoded = layers.Dense(16, activation="relu")(encoded)

    decoded = layers.Dense(32, activation="relu")(encoded)
    decoded = layers.Dense(64, activation="relu")(decoded)
    decoded = layers.Dropout(0.2)(decoded)
    decoded = layers.Dense(input_dim, activation="linear")(decoded)

    autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer="adam", loss="mse")

    checkpoint = ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = autoencoder.fit(
        X_train, X_train,
        epochs=100,
        batch_size=64,
        shuffle=True,
        validation_data=(X_val, X_val),
        callbacks=[checkpoint, early_stop],
        verbose=0
    )

    autoencoder.load_weights(MODEL_PATH)
    reconstructions = autoencoder.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)

    precisions, recalls, thresholds = precision_recall_curve(y, mse)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]

    np.save(THRESHOLD_PATH, best_threshold)

    return {
        "message": "Autoencoder trained successfully.",
        "best_f1_threshold": float(best_threshold),
        "best_f1_score": float(np.max(f1_scores))
    }

def predict_autoencoder_anomalies():
    X_scaled, y_true, _ = load_data()
    if not os.path.exists(MODEL_PATH) or not os.path.exists(THRESHOLD_PATH):
        return {"error": "Trained model or threshold not found. Please train the autoencoder first."}

    best_threshold = np.load(THRESHOLD_PATH)
    input_dim = X_scaled.shape[1]

    autoencoder = tf.keras.models.load_model(MODEL_PATH, compile=False)
    reconstructions = autoencoder.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
    predictions = (mse > best_threshold).astype(int)

    df = pd.read_csv("creditcard.csv")
    X = df.drop(columns=["Class"])
    result = X.copy()
    result["Anomaly"] = predictions
    result["Actual"] = y_true
    result["Reconstruction_Error"] = mse

    return result.to_dict(orient="records")
