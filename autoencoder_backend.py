# autoencoder.py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import joblib
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

def train_autoencoder():
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
    autoencoder.compile(optimizer="adam", loss="mean_squared_error")

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

def predict_autoencoder():
    df = pd.read_csv("creditcard.csv")
    model = tf.keras.models.load_model("models/autoencoder.h5", compile=False)
    scaler = joblib.load("models/scaler.pkl")

    X = df.drop(columns=["Class"])
    y_true = df["Class"].values
    X_scaled = scaler.transform(X)

    # Reconstruction
    X_pred = model.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)

    # Load threshold
    with open("models/threshold.txt", "r") as f:
        threshold = float(f.read())

    df["MSE"] = mse
    df["is_anomaly"] = df["MSE"] > threshold

    # Evaluation
    y_pred = df["is_anomaly"].astype(int).values
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Return both stats and top 100 anomalies
    top_anomalies = df[df["is_anomaly"] == 1].head(100).to_dict(orient="records")

    return {
        "anomalies": top_anomalies,
        "stats": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }
    }