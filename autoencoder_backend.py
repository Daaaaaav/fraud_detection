import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# File paths
MODEL_PATH = "models/autoencoder.h5"
RF_PATH = "models/random_forest.pkl"
BOTTLENECK_MODEL_PATH = "models/encoder.h5"
SCALER_PATH = "models/scaler.pkl"
DATA_PATH = "creditcard.csv"

def load_dataset():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values
    return X, y

def preprocess_data():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    return X_train_scaled, y_train, X_test_scaled, y_test

def build_autoencoder(input_dim):
    input_layer = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(input_layer)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    bottleneck = layers.Dense(16, activation="relu", name="bottleneck")(x)
    x = layers.Dense(32, activation="relu")(bottleneck)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    output_layer = layers.Dense(input_dim, activation="linear")(x)

    autoencoder = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    encoder = tf.keras.Model(inputs=input_layer, outputs=bottleneck)
    autoencoder.compile(optimizer=Adam(1e-4), loss="mse")

    return autoencoder, encoder

def compute_reconstruction_error(original, reconstructed):
    return np.mean(np.square(original - reconstructed), axis=1)

def augment_with_error(encoded, error_vector):
    return np.hstack((encoded, error_vector.reshape(-1, 1)))

def load_models():
    encoder = tf.keras.models.load_model(BOTTLENECK_MODEL_PATH, compile=False)
    autoencoder = tf.keras.models.load_model(MODEL_PATH, compile=False)
    rf_clf = joblib.load(RF_PATH)
    scaler = joblib.load(SCALER_PATH)
    return encoder, autoencoder, rf_clf, scaler


### --- Training Function --- ###

def train_autoencoder():
    X_train_scaled, y_train, X_test_scaled, y_test = preprocess_data()

    X_train_normal = X_train_scaled[y_train == 0]
    input_dim = X_train_scaled.shape[1]

    autoencoder, encoder = build_autoencoder(input_dim)

    # Training callbacks
    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    ]

    # Train autoencoder
    autoencoder.fit(
        X_train_normal, X_train_normal,
        validation_split=0.2,
        epochs=100,
        batch_size=256,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    autoencoder.load_weights(MODEL_PATH)

    # Train Random Forest on encoded features + reconstruction error
    encoded_train = encoder.predict(X_train_scaled, verbose=0)
    reconstructed_train = autoencoder.predict(X_train_scaled, verbose=0)
    errors_train = compute_reconstruction_error(X_train_scaled, reconstructed_train)
    features_train = augment_with_error(encoded_train, errors_train)

    X_balanced, y_balanced = SMOTE(random_state=42).fit_resample(features_train, y_train)

    rf_clf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
    rf_clf.fit(X_balanced, y_balanced)

    joblib.dump(rf_clf, RF_PATH)
    encoder.save(BOTTLENECK_MODEL_PATH)

    # Evaluation
    encoded_test = encoder.predict(X_test_scaled, verbose=0)
    reconstructed_test = autoencoder.predict(X_test_scaled, verbose=0)
    errors_test = compute_reconstruction_error(X_test_scaled, reconstructed_test)
    features_test = augment_with_error(encoded_test, errors_test)

    preds = rf_clf.predict(features_test)

    return {
        "message": "Autoencoder + RF trained successfully.",
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1_score": round(f1_score(y_test, preds), 4),
        "anomaly_rate": round(np.mean(preds), 4)
    }


### --- Prediction Functions --- ###

def predict_autoencoder():
    X, y = load_dataset()
    encoder, autoencoder, rf_clf, scaler = load_models()
    X_scaled = scaler.transform(X)

    bottleneck = encoder.predict(X_scaled, verbose=0)
    reconstructed = autoencoder.predict(X_scaled, verbose=0)
    error = compute_reconstruction_error(X_scaled, reconstructed)
    features = augment_with_error(bottleneck, error)

    preds = rf_clf.predict(features)
    df = pd.read_csv(DATA_PATH)
    df["Predicted"] = preds

    # Get samples for preview
    true_pos = df[(df["Class"] == 1) & (df["Predicted"] == 1)]
    true_neg = df[(df["Class"] == 0) & (df["Predicted"] == 0)]

    sample_tp = true_pos.sample(n=min(3, len(true_pos)), random_state=42)
    sample_tn = true_neg.sample(n=min(2, len(true_neg)), random_state=42)

    sample = pd.concat([sample_tp, sample_tn]).sample(frac=1, random_state=99)
    return sample.to_dict(orient="records")


def predict_autoencoder_manual(user_input):
    try:
        logging.info("Manual prediction using Autoencoder + RF")

        encoder, autoencoder, rf_clf, scaler = load_models()

        input_array = np.array(user_input).reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        bottleneck = encoder.predict(scaled_input, verbose=0)
        reconstructed = autoencoder.predict(scaled_input, verbose=0)
        error = compute_reconstruction_error(scaled_input, reconstructed).reshape(-1, 1)

        features = augment_with_error(bottleneck, error)
        prediction = rf_clf.predict(features)[0]

        return {"is_fraud": int(prediction)}

    except Exception as e:
        logging.exception("Error in predict_autoencoder_manual")
        return {"error": str(e)}