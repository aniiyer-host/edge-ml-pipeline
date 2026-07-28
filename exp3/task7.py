from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import polars as pl
import numpy as np


from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.metrics import (
    accuracy_score,
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

# ----------------------------------------------------
# Use a subset for faster hyperparameter tuning
# ----------------------------------------------------

X_subset, _, y_subset, _ = train_test_split(
    X_train,
    y_train,
    train_size=50000,
    stratify=y_train,
    random_state=42
)

# ====================================================
# KNN Hyperparameter Tuning
# ====================================================

print("="*70)
print("KNN Grid Search")
print("="*70)

knn_params = {
    "n_neighbors": [3, 5, 7, 9]
}

knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    knn_params,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

knn_grid.fit(X_subset, y_subset)

print("Best Parameters:", knn_grid.best_params_)
print("Best CV Accuracy:", knn_grid.best_score_)

best_knn = knn_grid.best_estimator_

knn_accuracy = accuracy_score(
    y_test,
    best_knn.predict(X_test)
)

print("Test Accuracy:", knn_accuracy)

# ====================================================
# Random Forest Hyperparameter Tuning
# ====================================================

print("\n" + "="*70)
print("Random Forest Grid Search")
print("="*70)

rf_params = {
    "n_estimators": [20, 30, 50],
    "max_depth": [10, 20, None]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ),
    rf_params,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

rf_grid.fit(X_subset, y_subset)

print("Best Parameters:", rf_grid.best_params_)
print("Best CV Accuracy:", rf_grid.best_score_)

best_rf = rf_grid.best_estimator_

rf_accuracy = accuracy_score(
    y_test,
    best_rf.predict(X_test)
)

print("Test Accuracy:", rf_accuracy)