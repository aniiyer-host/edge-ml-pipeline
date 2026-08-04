import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from scikeras.wrappers import KerasClassifier

from sklearn.model_selection import GridSearchCV

# ==========================================================
# Task 7 - Hyperparameter Tuning using Grid Search
# ==========================================================

print("=" * 60)
print("TASK 7 - HYPERPARAMETER TUNING")
print("=" * 60)

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")

y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# ----------------------------------------------------------
# Build Model Function
# ----------------------------------------------------------

def create_model(neurons1=16, neurons2=8):

    model = Sequential()

    model.add(Dense(
        neurons1,
        activation="relu",
        input_shape=(4,)
    ))

    model.add(Dense(
        neurons2,
        activation="relu"
    ))

    model.add(Dense(
        3,
        activation="softmax"
    ))

    model.compile(

        optimizer="adam",

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    return model

# ----------------------------------------------------------
# Wrap Model
# ----------------------------------------------------------

classifier = KerasClassifier(

    model=create_model,

    verbose=0

)

# ----------------------------------------------------------
# Hyperparameter Grid
# ----------------------------------------------------------

param_grid = {

    "model__neurons1": [16, 32],

    "model__neurons2": [8, 16],

    "batch_size": [8, 16],

    "epochs": [50, 75]

}

# ----------------------------------------------------------
# Grid Search
# ----------------------------------------------------------

grid = GridSearchCV(

    estimator=classifier,

    param_grid=param_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1

)

print("\nPerforming Grid Search...\n")

grid_result = grid.fit(

    X_train,

    y_train

)

# ----------------------------------------------------------
# Best Parameters
# ----------------------------------------------------------

print("=" * 60)
print("GRID SEARCH RESULTS")
print("=" * 60)

print("\nBest Cross Validation Accuracy")

print(f"{grid_result.best_score_:.4f}")

print("\nBest Parameters")

print(grid_result.best_params_)

# ----------------------------------------------------------
# Evaluate Best Model
# ----------------------------------------------------------

best_model = grid_result.best_estimator_

test_accuracy = best_model.score(

    X_test,

    y_test

)

print("\nTest Accuracy")

print(f"{test_accuracy:.4f}")

# ----------------------------------------------------------
# Save All Results
# ----------------------------------------------------------

results = pd.DataFrame(

    grid_result.cv_results_

)

results.to_csv(

    "GridSearch_Results.csv",

    index=False

)

# ----------------------------------------------------------
# Summary Table
# ----------------------------------------------------------

summary = pd.DataFrame({

    "Metric":[

        "Original Accuracy",

        "Best Cross Validation Accuracy",

        "Test Accuracy"

    ],

    "Value":[

        0.9667,

        round(grid_result.best_score_,4),

        round(test_accuracy,4)

    ]

})

summary.to_csv(

    "Hyperparameter_Tuning_Summary.csv",

    index=False

)

print("\nResults saved successfully.")

print("\nTask 7 Completed Successfully.")