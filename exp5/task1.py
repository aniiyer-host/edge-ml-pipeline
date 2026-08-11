import os
import time
import numpy as np
import tensorflow as tf

# ==========================================================
# Experiment 5 - Task 1
# Baseline Model Preparation
# ==========================================================

print("=" * 60)
print("EXPERIMENT 5 - TASK 1")
print("Baseline Model Preparation")
print("=" * 60)

# ----------------------------------------------------------
# File Names
# ----------------------------------------------------------

MODEL_NAME = "TinyNN_Iris.keras"

# ----------------------------------------------------------
# Check Required Files
# ----------------------------------------------------------

required_files = [
    "X_train.npy",
    "X_test.npy",
    "y_train.npy",
    "y_test.npy",
    MODEL_NAME
]

for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(
            f"Required file '{file}' was not found."
        )

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")

y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print("\nDataset Information")
print("-" * 60)
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
print(f"Input features   : {X_train.shape[1]}")
print(f"Number of classes: {len(np.unique(y_train))}")

# ----------------------------------------------------------
# Load Existing Tiny Neural Network
# ----------------------------------------------------------

print("\nLoading Tiny Neural Network...")

model = tf.keras.models.load_model(MODEL_NAME)

print("Model loaded successfully.")

# ----------------------------------------------------------
# Display Model Architecture
# ----------------------------------------------------------

print("\nModel Architecture")
print("-" * 60)

model.summary()

print(f"\nTrainable Parameters : {model.count_params()}")

# ----------------------------------------------------------
# Measure Model Size
# ----------------------------------------------------------

model_size_kb = os.path.getsize(MODEL_NAME) / 1024

# ----------------------------------------------------------
# Evaluate Baseline Accuracy
# ----------------------------------------------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

# ----------------------------------------------------------
# Measure Inference Time
# ----------------------------------------------------------
# Run one warm-up inference first so that initialization
# overhead does not affect the timing measurement.

model.predict(X_test[:1], verbose=0)

start_time = time.perf_counter()

predictions = model.predict(
    X_test,
    verbose=0
)

end_time = time.perf_counter()

total_inference_time = end_time - start_time

# Average inference time per sample in milliseconds
inference_time_ms = (
    total_inference_time / len(X_test)
) * 1000

# ----------------------------------------------------------
# Display Results
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("BASELINE MODEL RESULTS")
print("=" * 60)

print(f"Model File           : {MODEL_NAME}")
print(f"Model Size           : {model_size_kb:.2f} KB")
print(f"Test Accuracy        : {accuracy * 100:.2f}%")
print(f"Test Loss            : {loss:.4f}")
print(f"Total Inference Time : {total_inference_time * 1000:.4f} ms")
print(f"Average Inference    : {inference_time_ms:.4f} ms/sample")

print("=" * 60)
print("Task 1 Completed Successfully.")
print("=" * 60)
