import os
import time
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ==========================================================
# Task 3 - Tiny Neural Network
# ==========================================================

# -----------------------------
# Load Dataset
# -----------------------------
X_train = np.load("X_train.npy")

X_test = np.load("X_test.npy")

y_train = np.load("y_train.npy")

y_test = np.load("y_test.npy")

print("=" * 60)
print("Training Tiny Neural Network")
print("=" * 60)

# -----------------------------
# Build Model
# -----------------------------
model = Sequential([

    Dense(
        16,
        activation="relu",
        input_shape=(4,)
    ),

    Dense(
        8,
        activation="relu"
    ),

    Dense(
        3,
        activation="softmax"
    )

])

# -----------------------------
# Compile
# -----------------------------
model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

print(model.summary())

# -----------------------------
# Parameters
# -----------------------------
print("\nTrainable Parameters :", model.count_params())

# -----------------------------
# Train
# -----------------------------
start = time.time()

history = model.fit(

    X_train,

    y_train,

    validation_data=(X_test, y_test),

    epochs=50,

    batch_size=8,

    verbose=1

)

training_time = time.time() - start

# -----------------------------
# Save Model
# -----------------------------
MODEL_NAME = "TinyNN_Iris.keras"

model.save(MODEL_NAME)

model_size = os.path.getsize(MODEL_NAME) / 1024

# -----------------------------
# Save History
# -----------------------------
np.save("train_accuracy.npy", history.history["accuracy"])

np.save("val_accuracy.npy", history.history["val_accuracy"])

np.save("train_loss.npy", history.history["loss"])

np.save("val_loss.npy", history.history["val_loss"])

# -----------------------------
# Evaluate
# -----------------------------
loss, accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=0

)

print("\n" + "=" * 60)
print("MODEL SUMMARY")
print("=" * 60)

print(f"Training Time : {training_time:.2f} seconds")

print(f"Model Size    : {model_size:.2f} KB")

print(f"Test Accuracy : {accuracy:.4f}")

print(f"Test Loss     : {loss:.4f}")

print("\nTask 3 Completed Successfully.")