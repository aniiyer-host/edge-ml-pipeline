import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==========================================================
# Task 2 - Data Preprocessing
# ==========================================================

df = pd.read_csv("Iris.csv")

# -----------------------------
# Features & Labels
# -----------------------------
X = df.drop(columns=["Id", "Species"])

y = df["Species"]

# -----------------------------
# Label Encoding
# -----------------------------
encoder = LabelEncoder()

y = encoder.fit_transform(y)

print("=" * 60)
print("Encoded Classes")
print("=" * 60)

for i, cls in enumerate(encoder.classes_):
    print(i, "->", cls)

# -----------------------------
# Feature Normalization
# -----------------------------
scaler = StandardScaler()

X = scaler.fit_transform(X)

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Samples :", X_train.shape[0])

print("Testing Samples  :", X_test.shape[0])

print("Training Features :", X_train.shape)

print("Testing Features  :", X_test.shape)

# -----------------------------
# Save for Next Task
# -----------------------------
np.save("X_train.npy", X_train)

np.save("X_test.npy", X_test)

np.save("y_train.npy", y_train)

np.save("y_test.npy", y_test)

print("\nPreprocessed data saved successfully.")

print("\nTask 2 Completed Successfully.")