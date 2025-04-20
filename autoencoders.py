import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os

df = pd.read_csv("creditcard.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_normal = X_scaled[y == 0]
X_fraud = X_scaled[y == 1]

X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)
X_test = X_scaled
y_test = y.values

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

autoencoder.summary()

checkpoint_path = "best_autoencoder.h5"
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss',
                                   save_best_only=True, verbose=1)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = autoencoder.fit(
    X_train, X_train,
    epochs=100,
    batch_size=64,
    shuffle=True,
    validation_data=(X_val, X_val),
    callbacks=[model_checkpoint, early_stopping]
)

autoencoder.load_weights(checkpoint_path)
print("Loaded saved model:", checkpoint_path)

reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

precisions, recalls, thresholds = precision_recall_curve(y_test, mse)

f1_scores = np.divide(2 * (precisions[:-1] * recalls[:-1]),
                      (precisions[:-1] + recalls[:-1]),
                      out=np.zeros_like(precisions[:-1]),
                      where=(precisions[:-1] + recalls[:-1]) != 0)

best_threshold = thresholds[np.argmax(f1_scores)]
best_f1 = np.max(f1_scores)

print(f"Best F1 Threshold: {best_threshold:.4f} | Best F1 Score: {best_f1:.4f}")

predictions = (mse > best_threshold).astype(int)

print("\n Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\n Classification Report:\n", classification_report(y_test, predictions, digits=4))

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Autoencoder Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()

plt.plot(recalls[:-1], precisions[:-1])
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()