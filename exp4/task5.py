import os
import time
import numpy as np
import tensorflow as tf
import pandas as pd

# ==========================================================
# Task 5 - Performance Comparison
# ==========================================================

print("=" * 60)
print("TASK 5 - PERFORMANCE COMPARISON")
print("=" * 60)

# ----------------------------------------------------------
# Load Evaluation Metrics
# ----------------------------------------------------------

metrics = np.load("evaluation_metrics.npy", allow_pickle=True).item()

accuracy = metrics["accuracy"]
precision = metrics["precision"]
recall = metrics["recall"]
f1 = metrics["f1"]
prediction_time = metrics["prediction_time"]

# ----------------------------------------------------------
# Load Model
# ----------------------------------------------------------

model = tf.keras.models.load_model("TinyNN_Iris.keras")

# ----------------------------------------------------------
# Load Test Data
# ----------------------------------------------------------

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# ----------------------------------------------------------
# Measure Inference Time
# ----------------------------------------------------------

start = time.time()

_ = model.predict(X_test, verbose=0)

end = time.time()

prediction_time = end - start

# ----------------------------------------------------------
# Model Information
# ----------------------------------------------------------

parameters = model.count_params()

model_size = os.path.getsize("TinyNN_Iris.keras") / 1024

memory_usage = model_size

# Training time obtained from Task 3
training_time = 3.94

# ----------------------------------------------------------
# Performance Table
# ----------------------------------------------------------

performance = pd.DataFrame({

    "Parameter":[

        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Training Time (s)",
        "Prediction Time (s)",
        "Model Size (KB)",
        "Memory Usage (KB)",
        "Trainable Parameters"

    ],

    "Tiny Neural Network":[

        round(accuracy,4),
        round(precision,4),
        round(recall,4),
        round(f1,4),
        round(training_time,4),
        round(prediction_time,6),
        round(model_size,2),
        round(memory_usage,2),
        parameters

    ]

})

print("\nPerformance Comparison\n")

print(performance)

# ----------------------------------------------------------
# Save Table
# ----------------------------------------------------------

performance.to_csv(

    "Performance_Comparison.csv",

    index=False

)

print("\nPerformance table saved as Performance_Comparison.csv")

# ----------------------------------------------------------
# Best Model
# ----------------------------------------------------------

print("\nBest Performing Model")

print("----------------------")

print("Tiny Neural Network")

print(f"Accuracy : {accuracy:.4f}")

print("\nReason")

print("The Tiny Neural Network provides high classification accuracy")
print("while maintaining a compact model size and low inference time,")
print("making it highly suitable for Edge AI deployment.")

print("\nTask 5 Completed Successfully.")