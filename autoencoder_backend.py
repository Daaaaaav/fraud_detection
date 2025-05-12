import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

MODEL_PATH = "models/autoencoder.h5"
RF_PATH = "models/random_forest.pkl"
BOTTLENECK_MODEL_PATH = "models/encoder.h5"
SCALER_PATH = "models/scaler.pkl"
THRESHOLD_PATH = "models/threshold.txt"

def load_data():
    df = pd.read_csv("creditcard.csv")
    X = df.drop(columns=["Class"])
    y = df["Class"].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def train_autoencoder():
    X_scaled, y, scaler = load_data()
    joblib.dump(scaler, SCALER_PATH)

    X_normal = X_scaled[y == 0]
    X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)
    input_dim = X_train.shape[1]

    # Define autoencoder and encoder
    input_layer = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation="relu")(x)
    bottleneck = layers.Dense(16, activation="relu", name="bottleneck")(x)
    x = layers.Dense(32, activation="relu")(bottleneck)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    output_layer = layers.Dense(input_dim, activation="linear")(x)

    autoencoder = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=Adam(learning_rate=1e-4), loss="mean_absolute_error")
    encoder = tf.keras.Model(inputs=input_layer, outputs=bottleneck)

    # Train autoencoder
    checkpoint = ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    autoencoder.fit(
        X_train, X_train,
        epochs=200,
        batch_size=256,
        shuffle=True,
        validation_data=(X_val, X_val),
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )

    autoencoder.load_weights(MODEL_PATH)

    # Extract bottleneck features
    bottleneck_features = encoder.predict(X_scaled, verbose=0)

    # Train Random Forest on encoded features
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(bottleneck_features, y)

    joblib.dump(rf_clf, RF_PATH)
    encoder.save(BOTTLENECK_MODEL_PATH)

    # Evaluate
    preds = rf_clf.predict(bottleneck_features)
    accuracy = accuracy_score(y, preds)
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)
    anomaly_rate = np.mean(preds)

    return {
        "message": "Autoencoder model with Random Forest base trained successfully.",
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "anomaly_rate": round(anomaly_rate, 4)
    }

def predict_autoencoder():
    df = pd.read_csv("creditcard.csv")
    model = tf.keras.models.load_model(BOTTLENECK_MODEL_PATH, compile=False)
    rf_clf = joblib.load(RF_PATH)
    scaler = joblib.load(SCALER_PATH)

    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_scaled = scaler.transform(X)

    bottleneck_features = model.predict(X_scaled, verbose=0)
    predictions = rf_clf.predict(bottleneck_features)
    df["Predicted"] = predictions

    true_positives = df[(df["Class"] == 1) & (df["Predicted"] == 1)]
    true_negatives = df[(df["Class"] == 0) & (df["Predicted"] == 0)]

    tp_sample = true_positives.sample(n=min(3, len(true_positives)), random_state=42)
    tn_sample = true_negatives.sample(n=min(2, len(true_negatives)), random_state=42)

    combined_sample = pd.concat([tp_sample, tn_sample]).sample(frac=1, random_state=99)
    return combined_sample.to_dict(orient="records")
