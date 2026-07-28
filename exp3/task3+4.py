import polars as pl
import numpy as np
import time
import pickle
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# -----------------------------
# Load Dataset
# -----------------------------
train_df = pl.read_csv("train.csv")
test_df = pl.read_csv("test.csv")

label_column = "Activity"     

X_train = train_df.drop([label_column, "timestamp"]).to_numpy()
X_test = test_df.drop([label_column, "timestamp"]).to_numpy()

y_train = train_df[label_column].to_numpy()
y_test = test_df[label_column].to_numpy()

# Encode Labels
encoder = LabelEncoder()

y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# -----------------------------
# Models
# -----------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        random_state=42
    ),

    "K-Nearest Neighbour": KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=20
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=30,      # 100 -> 30
        max_depth=20,         # Prevents huge trees
        random_state=42,
        n_jobs=-1             # Uses all CPU cores
    ),

    "Support Vector Machine": LinearSVC(
        random_state=42,
        max_iter=5000
    ),

    "Gaussian Naive Bayes": GaussianNB()
}

print("-"*100)
print(f"{'Model':30} {'Train Time(s)':15} {'Model Size(KB)':15} {'Parameters':15} {'Accuracy'}")
print("-"*100)

results = {}

for name, model in models.items():

    # -----------------------------
    # Train Model
    # -----------------------------
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    # -----------------------------
    # Predictions
    # -----------------------------
    pred_start = time.perf_counter()
    y_pred = model.predict(X_test)
    prediction_time = time.perf_counter() - pred_start

    # -----------------------------
    # Task 3
    # -----------------------------

    accuracy = accuracy_score(y_test, y_pred)

    # Calculate model size
    filename = "temp_model.pkl"
    with open(filename, "wb") as f:
        pickle.dump(model, f)

    model_size = os.path.getsize(filename) / 1024
    os.remove(filename)

    # Count parameters
    if hasattr(model, "coef_"):
        parameters = model.coef_.size

    elif hasattr(model, "tree_"):
        parameters = model.tree_.node_count

    elif hasattr(model, "estimators_"):
        parameters = sum(tree.tree_.node_count for tree in model.estimators_)

    else:
        parameters = "N/A"

    results[name] = accuracy

    print(f"{name:30} {train_time:<15.4f} {model_size:<15.2f} {str(parameters):15} {accuracy:.4f}")
    print(f"Prediction Time: {prediction_time:.4f} s")

    # -----------------------------
    # Task 4
    # -----------------------------

    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n" + "="*70)
    print(name)
    print("="*70)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report")
    print(classification_report(
        y_test,
        y_pred,
        target_names=[str(x) for x in encoder.classes_],
        zero_division=0
    ))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=encoder.classes_
    )

    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.show()

# -----------------------------
# Accuracy Comparison Chart
# -----------------------------

plt.figure(figsize=(10,5))
plt.bar(results.keys(), results.values())
plt.ylabel("Accuracy")
plt.xlabel("Models")
plt.title("Accuracy Comparison of Classifiers")
plt.xticks(rotation=20)
plt.ylim(0,1)
plt.tight_layout()
plt.show()
