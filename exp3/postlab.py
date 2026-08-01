import polars as pl
import joblib
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

train_df = pl.read_csv("train.csv")

label_column = "Activity"

# Features and Labels
X_train = train_df.drop([label_column, "timestamp"]).to_numpy()
y_train = train_df[label_column].to_numpy()

# Encode Labels
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)

# --------------------------------------------------
# Train Best Model
# --------------------------------------------------

best_model = GaussianNB()

best_model.fit(X_train, y_train)

# --------------------------------------------------
# Export Model using Joblib
# --------------------------------------------------

model_filename = "gaussian_nb_model.joblib"

joblib.dump(best_model, model_filename)

print("Model exported successfully.")
print("Filename:", model_filename)

# --------------------------------------------------
# Estimate Storage Requirement
# --------------------------------------------------

model_size_bytes = os.path.getsize(model_filename)
model_size_kb = model_size_bytes / 1024
model_size_mb = model_size_kb / 1024

print("\nModel Storage Requirement")
print("-" * 40)
print(f"Bytes : {model_size_bytes}")
print(f"KB    : {model_size_kb:.2f}")
print(f"MB    : {model_size_mb:.4f}")

# --------------------------------------------------
# Deployment Suitability
# --------------------------------------------------

print("\nDeployment Suitability")
print("-" * 40)

devices = {
    "Arduino Nano 33 BLE Sense": 1024,      # 1 MB
    "ESP32": 4096,                           # 4 MB
    "Raspberry Pi Pico": 2048,               # 2 MB
    "STM32": 1024                            # Typical 1 MB
}

for device, flash_kb in devices.items():

    if model_size_kb <= flash_kb:
        status = "Suitable"
    else:
        status = "Not Suitable"

    print(f"{device:30} Flash: {flash_kb:5} KB   -->   {status}")