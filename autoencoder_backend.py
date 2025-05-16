import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import logging
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight

MODEL_PATH = "models/autoencoder.h5"
RF_PATH = "models/random_forest.pkl"
BOTTLENECK_MODEL_PATH = "models/encoder.h5"
SCALER_PATH = "models/scaler.pkl"

def load_data():
    df = pd.read_csv("creditcard.csv")
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = MinMaxScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PATH)

    return X_train_full_scaled, y_train_full, X_test_scaled, y_test, scaler


from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE

def train_autoencoder():
    X_train_full_scaled, y_train_full, X_test_scaled, y_test, _ = load_data()

    X_normal = X_train_full_scaled[y_train_full == 0]

    input_dim = X_train_full_scaled.shape[1]
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
    autoencoder.compile(optimizer=Adam(learning_rate=1e-4), loss="mse")
    encoder = tf.keras.Model(inputs=input_layer, outputs=bottleneck)

    checkpoint = ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    autoencoder.fit(
        X_normal, X_normal,
        epochs=100,
        batch_size=256,
        shuffle=True,
        validation_split=0.2,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )

    autoencoder.load_weights(MODEL_PATH)

    encoded_features_train = encoder.predict(X_train_full_scaled, verbose=0)
    reconstructed_train = autoencoder.predict(X_train_full_scaled, verbose=0)

    reconstruction_errors = np.mean(np.square(X_train_full_scaled - reconstructed_train), axis=1)
    train_features_with_error = np.hstack((encoded_features_train, reconstruction_errors.reshape(-1, 1)))

    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(train_features_with_error, y_train_full)

    rf_clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42
    )
    rf_clf.fit(X_balanced, y_balanced)

    joblib.dump(rf_clf, RF_PATH)
    encoder.save(BOTTLENECK_MODEL_PATH)

    encoded_test = encoder.predict(X_test_scaled, verbose=0)
    reconstructed_test = autoencoder.predict(X_test_scaled, verbose=0)
    reconstruction_errors_test = np.mean(np.square(X_test_scaled - reconstructed_test), axis=1)
    test_features_with_error = np.hstack((encoded_test, reconstruction_errors_test.reshape(-1, 1)))

    preds = rf_clf.predict(test_features_with_error)

    return {
        "message": "Autoencoder + RF with reconstruction error + SMOTE trained.",
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1_score": round(f1_score(y_test, preds), 4),
        "anomaly_rate": round(np.mean(preds), 4)
    }

def predict_autoencoder():
    df = pd.read_csv("creditcard.csv")
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    encoder = tf.keras.models.load_model(BOTTLENECK_MODEL_PATH, compile=False)
    autoencoder = tf.keras.models.load_model(MODEL_PATH, compile=False)
    rf_clf = joblib.load(RF_PATH)
    scaler = joblib.load(SCALER_PATH)

    X_scaled = scaler.transform(X)

    bottleneck_features = encoder.predict(X_scaled, verbose=0)
    reconstructed = autoencoder.predict(X_scaled, verbose=0)
    reconstruction_errors = np.mean(np.square(X_scaled - reconstructed), axis=1)

    features_with_error = np.hstack((bottleneck_features, reconstruction_errors.reshape(-1, 1)))
    predictions = rf_clf.predict(features_with_error)

    df["Predicted"] = predictions

    true_positives = df[(df["Class"] == 1) & (df["Predicted"] == 1)]
    true_negatives = df[(df["Class"] == 0) & (df["Predicted"] == 0)]

    tp_sample = true_positives.sample(n=min(3, len(true_positives)), random_state=42)
    tn_sample = true_negatives.sample(n=min(2, len(true_negatives)), random_state=42)

    combined_sample = pd.concat([tp_sample, tn_sample]).sample(frac=1, random_state=99)
    return combined_sample.to_dict(orient="records")


def predict_autoencoder_manual(user_input):
    try:
        logging.info("Predicting using encoder + reconstruction error + Random Forest")

        scaler = joblib.load(SCALER_PATH)
        encoder = tf.keras.models.load_model(BOTTLENECK_MODEL_PATH, compile=False)
        autoencoder = tf.keras.models.load_model(MODEL_PATH, compile=False)
        rf_clf = joblib.load(RF_PATH)

        input_array = np.array(user_input).reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        bottleneck_features = encoder.predict(scaled_input, verbose=0)
        reconstructed = autoencoder.predict(scaled_input, verbose=0)
        reconstruction_error = np.mean(np.square(scaled_input - reconstructed), axis=1).reshape(-1, 1)

        features_with_error = np.hstack((bottleneck_features, reconstruction_error))

        prediction = rf_clf.predict(features_with_error)[0]

        return {
            "is_fraud": int(prediction)
        }

    except Exception as e:
        logging.exception("Error in predict_autoencoder_manual")
        return {"error": str(e)}

