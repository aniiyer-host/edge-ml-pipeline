import os
import time
import pickle
import tracemalloc
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# ==========================================================
# MODEL COMPARISON - IRIS DATASET
# ==========================================================

print("=" * 70)
print("MODEL COMPARISON - IRIS DATASET")
print("=" * 70)


# ==========================================================
# LOAD DATA
# ==========================================================

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")

y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print("\nDataset Loaded")

print("Training Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])
print("Features         :", X_train.shape[1])


# ==========================================================
# RESULTS STORAGE
# ==========================================================

results = []


# ==========================================================
# FUNCTION FOR CLASSICAL ML MODELS
# ==========================================================

def evaluate_sklearn_model(name, model):

    print("\n" + "-" * 70)
    print("Training:", name)
    print("-" * 70)

    # -----------------------------
    # Training Memory
    # -----------------------------

    tracemalloc.start()

    start_train = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_train

    current, peak_memory = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    # -----------------------------
    # Prediction
    # -----------------------------

    start_prediction = time.perf_counter()

    y_pred = model.predict(X_test)

    prediction_time = time.perf_counter() - start_prediction

    # -----------------------------
    # Metrics
    # -----------------------------

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

    # -----------------------------
    # Model Size
    # -----------------------------

    temp_file = f"temp_{name.replace(' ', '_')}.pkl"

    with open(temp_file, "wb") as f:
        pickle.dump(model, f)

    model_size = os.path.getsize(temp_file) / 1024

    os.remove(temp_file)

    # -----------------------------
    # Parameters
    # -----------------------------

    try:

        if hasattr(model, "coef_"):

            parameters = model.coef_.size + model.intercept_.size

        elif hasattr(model, "tree_"):

            parameters = model.tree_.node_count

        elif hasattr(model, "estimators_"):

            parameters = sum(
                estimator.tree_.node_count
                for estimator in model.estimators_
            )

        elif hasattr(model, "_fit_X"):

            parameters = model._fit_X.size

        else:

            parameters = "N/A"

    except:

        parameters = "N/A"

    # -----------------------------
    # Print Results
    # -----------------------------

    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print(f"Training Time  : {training_time:.6f} s")
    print(f"Prediction Time: {prediction_time:.6f} s")
    print(f"Memory Usage   : {peak_memory / 1024:.2f} KB")
    print(f"Model Size     : {model_size:.2f} KB")
    print(f"Parameters     : {parameters}")

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "Training Time (s)": training_time,

        "Prediction Time (s)": prediction_time,

        "Memory Usage (KB)": peak_memory / 1024,

        "Model Size (KB)": model_size,

        "Parameters": parameters

    })


# ==========================================================
# 1. LOGISTIC REGRESSION
# ==========================================================

lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

evaluate_sklearn_model(
    "Logistic Regression",
    lr
)


# ==========================================================
# 2. K-NEAREST NEIGHBOUR
# ==========================================================

knn = KNeighborsClassifier(
    n_neighbors=5
)

evaluate_sklearn_model(
    "KNN",
    knn
)


# ==========================================================
# 3. DECISION TREE
# ==========================================================

dt = DecisionTreeClassifier(
    random_state=42
)

evaluate_sklearn_model(
    "Decision Tree",
    dt
)


# ==========================================================
# 4. RANDOM FOREST
# ==========================================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

evaluate_sklearn_model(
    "Random Forest",
    rf
)


# ==========================================================
# 5. SUPPORT VECTOR MACHINE
# ==========================================================

svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)

evaluate_sklearn_model(
    "SVM",
    svm
)


# ==========================================================
# 6. GAUSSIAN NAIVE BAYES
# ==========================================================

nb = GaussianNB()

evaluate_sklearn_model(
    "Naive Bayes",
    nb
)


# ==========================================================
# 7. REGULAR NEURAL NETWORK
# ==========================================================

print("\n" + "-" * 70)
print("Training: Regular Neural Network")
print("-" * 70)


regular_nn = Sequential([

    Dense(
        64,
        activation="relu",
        input_shape=(X_train.shape[1],)
    ),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        3,
        activation="softmax"
    )

])


regular_nn.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


tracemalloc.start()

start_train = time.perf_counter()

regular_nn.fit(

    X_train,
    y_train,

    validation_split=0.2,

    epochs=50,

    batch_size=8,

    verbose=0

)

regular_training_time = time.perf_counter() - start_train

current, regular_peak_memory = tracemalloc.get_traced_memory()

tracemalloc.stop()


start_prediction = time.perf_counter()

regular_predictions = regular_nn.predict(
    X_test,
    verbose=0
)

regular_prediction_time = (
    time.perf_counter() - start_prediction
)

regular_y_pred = np.argmax(
    regular_predictions,
    axis=1
)


regular_accuracy = accuracy_score(
    y_test,
    regular_y_pred
)

regular_precision = precision_score(
    y_test,
    regular_y_pred,
    average="weighted"
)

regular_recall = recall_score(
    y_test,
    regular_y_pred,
    average="weighted"
)

regular_f1 = f1_score(
    y_test,
    regular_y_pred,
    average="weighted"
)


regular_nn.save(
    "RegularNN_Iris.keras"
)

regular_model_size = (
    os.path.getsize("RegularNN_Iris.keras") / 1024
)

regular_parameters = regular_nn.count_params()


print(f"Accuracy       : {regular_accuracy:.4f}")
print(f"Precision      : {regular_precision:.4f}")
print(f"Recall         : {regular_recall:.4f}")
print(f"F1 Score       : {regular_f1:.4f}")
print(f"Training Time  : {regular_training_time:.6f} s")
print(f"Prediction Time: {regular_prediction_time:.6f} s")
print(f"Memory Usage   : {regular_peak_memory / 1024:.2f} KB")
print(f"Model Size     : {regular_model_size:.2f} KB")
print(f"Parameters     : {regular_parameters}")


results.append({

    "Model": "Regular Neural Network",

    "Accuracy": regular_accuracy,

    "Precision": regular_precision,

    "Recall": regular_recall,

    "F1 Score": regular_f1,

    "Training Time (s)": regular_training_time,

    "Prediction Time (s)": regular_prediction_time,

    "Memory Usage (KB)": regular_peak_memory / 1024,

    "Model Size (KB)": regular_model_size,

    "Parameters": regular_parameters

})


# ==========================================================
# 8. TINY NEURAL NETWORK
# ==========================================================

print("\n" + "-" * 70)
print("Evaluating: Tiny Neural Network")
print("-" * 70)


tiny_nn = tf.keras.models.load_model(
    "TinyNN_Iris.keras"
)


# ----------------------------------------------------------
# Load Existing Training Time
# ----------------------------------------------------------

tiny_training_time = 3.94


# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

tracemalloc.start()

start_prediction = time.perf_counter()

tiny_predictions = tiny_nn.predict(
    X_test,
    verbose=0
)

tiny_prediction_time = (
    time.perf_counter() - start_prediction
)

current, tiny_peak_memory = tracemalloc.get_traced_memory()

tracemalloc.stop()


tiny_y_pred = np.argmax(
    tiny_predictions,
    axis=1
)


# ----------------------------------------------------------
# Metrics
# ----------------------------------------------------------

tiny_accuracy = accuracy_score(
    y_test,
    tiny_y_pred
)

tiny_precision = precision_score(
    y_test,
    tiny_y_pred,
    average="weighted"
)

tiny_recall = recall_score(
    y_test,
    tiny_y_pred,
    average="weighted"
)

tiny_f1 = f1_score(
    y_test,
    tiny_y_pred,
    average="weighted"
)


tiny_model_size = (
    os.path.getsize("TinyNN_Iris.keras") / 1024
)

tiny_parameters = tiny_nn.count_params()


print(f"Accuracy       : {tiny_accuracy:.4f}")
print(f"Precision      : {tiny_precision:.4f}")
print(f"Recall         : {tiny_recall:.4f}")
print(f"F1 Score       : {tiny_f1:.4f}")
print(f"Training Time  : {tiny_training_time:.6f} s")
print(f"Prediction Time: {tiny_prediction_time:.6f} s")
print(f"Memory Usage   : {tiny_peak_memory / 1024:.2f} KB")
print(f"Model Size     : {tiny_model_size:.2f} KB")
print(f"Parameters     : {tiny_parameters}")


results.append({

    "Model": "Tiny Neural Network",

    "Accuracy": tiny_accuracy,

    "Precision": tiny_precision,

    "Recall": tiny_recall,

    "F1 Score": tiny_f1,

    "Training Time (s)": tiny_training_time,

    "Prediction Time (s)": tiny_prediction_time,

    "Memory Usage (KB)": tiny_peak_memory / 1024,

    "Model Size (KB)": tiny_model_size,

    "Parameters": tiny_parameters

})


# ==========================================================
# FINAL COMPARISON TABLE
# ==========================================================

results_df = pd.DataFrame(results)


print("\n")
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ==========================================================
# SAVE RESULTS
# ==========================================================

results_df.to_csv(
    "All_Model_Comparison.csv",
    index=False
)


print("\nResults saved to:")
print("All_Model_Comparison.csv")


# ==========================================================
# BEST MODEL
# ==========================================================

best_model = results_df.loc[
    results_df["Accuracy"].idxmax()
]

print("\n")
print("=" * 70)
print("BEST MODEL BASED ON ACCURACY")
print("=" * 70)

print(
    "Model   :",
    best_model["Model"]
)

print(
    "Accuracy:",
    f"{best_model['Accuracy'] * 100:.2f}%"
)


print("\nModel comparison completed successfully.")