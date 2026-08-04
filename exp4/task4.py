import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ==========================================================
# Task 4 - Model Evaluation
# ==========================================================

print("=" * 60)
print("TASK 4 - MODEL EVALUATION")
print("=" * 60)

# ----------------------------------------------------------
# Load Model
# ----------------------------------------------------------

model = tf.keras.models.load_model("TinyNN_Iris.keras")

# ----------------------------------------------------------
# Load Test Dataset
# ----------------------------------------------------------

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# ----------------------------------------------------------
# Load Training History
# ----------------------------------------------------------

train_accuracy = np.load("train_accuracy.npy")
val_accuracy = np.load("val_accuracy.npy")

train_loss = np.load("train_loss.npy")
val_loss = np.load("val_loss.npy")

# ----------------------------------------------------------
# Predictions
# ----------------------------------------------------------

prediction_start = tf.timestamp()

predictions = model.predict(X_test, verbose=0)

prediction_end = tf.timestamp()

prediction_time = float(prediction_end - prediction_start)

y_pred = np.argmax(predictions, axis=1)

# ----------------------------------------------------------
# Evaluation Metrics
# ----------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

cm = confusion_matrix(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Setosa",
        "Versicolor",
        "Virginica"
    ]
)

print("\nEvaluation Results")
print("-" * 40)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"Prediction Time : {prediction_time:.6f} seconds")

print("\nClassification Report\n")
print(report)

# ----------------------------------------------------------
# Plot Accuracy Curve
# ----------------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(train_accuracy, label="Training Accuracy")

plt.plot(val_accuracy, label="Validation Accuracy")

plt.title("Training vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# ----------------------------------------------------------
# Plot Loss Curve
# ----------------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(train_loss, label="Training Loss")

plt.plot(val_loss, label="Validation Loss")

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# ----------------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------------

plt.figure(figsize=(6,6))

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=[
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.show()

# ----------------------------------------------------------
# Accuracy Comparison Chart
# ----------------------------------------------------------

plt.figure(figsize=(6,5))

plt.bar(

    ["TinyNN"],

    [accuracy]

)

plt.ylim(0,1)

plt.ylabel("Accuracy")

plt.title("Model Accuracy")

for i,v in enumerate([accuracy]):

    plt.text(i,v+0.02,f"{v:.3f}",ha="center")

plt.tight_layout()

plt.show()

# ----------------------------------------------------------
# Save Metrics
# ----------------------------------------------------------

metrics = {

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1": f1,

    "prediction_time": prediction_time

}

np.save("evaluation_metrics.npy", metrics)

print("\nMetrics saved successfully.")

print("\nTask 4 Completed Successfully.")