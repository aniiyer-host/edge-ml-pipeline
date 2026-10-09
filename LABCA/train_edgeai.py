import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ---------------- Configuration ----------------
SEED = 42
WINDOW_SIZE = 50
N_FEATURES = 6
EPOCHS = 120
BATCH_SIZE = 8

LABELS = [
    "idle",
    "wave_left",
    "wave_right",
    "flick_up",
    "flick_down",
    "wrist_rotate",
]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}

CSV_PATH = Path("gesture-cmd/server/data/gesture_dataset_clean.csv")
OUT = Path("gesture-cmd/server/edgeai_v2")
DATA_OUT = OUT / "data"
MODEL_OUT = OUT / "models"

for directory in (DATA_OUT, MODEL_OUT):
    directory.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------- Load and validate ----------------
print("TensorFlow:", tf.__version__)
print("Loading:", CSV_PATH)

df = pd.read_csv(CSV_PATH)
required = [
    "label", "sample_id", "sample", "timestamp_ms",
    "ax", "ay", "az", "gx", "gy", "gz",
]
if list(df.columns) != required:
    raise ValueError(f"Unexpected CSV columns: {list(df.columns)}")

FEATURES = ["ax", "ay", "az", "gx", "gy", "gz"]
windows = []

for (label, sid), group in df.groupby(
    ["label", "sample_id"], sort=False
):
    group = group.sort_values("sample")

    if label not in LABEL_TO_ID:
        raise ValueError(f"Unknown label: {label}")
    if len(group) != WINDOW_SIZE:
        raise ValueError(f"Incomplete window: {label}/{sid}")
    if group["sample"].tolist() != list(range(WINDOW_SIZE)):
        raise ValueError(f"Bad sample indices: {label}/{sid}")

    values = group[FEATURES].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite sensor values: {label}/{sid}")

    windows.append({
        "label": label,
        "sample_id": int(sid),
        "x": values,
    })

if set(item["label"] for item in windows) != set(LABELS):
    raise ValueError("The dataset does not contain exactly the six expected labels.")

print(f"Loaded {len(windows)} complete windows.")

# ---------------- Split whole windows ----------------
# Deterministic 60/20/20 split independently within each class.
# No individual timesteps from a window can leak across splits.
rng = np.random.default_rng(SEED)
splits = {"train": [], "val": [], "test": []}

for label in LABELS:
    items = [w for w in windows if w["label"] == label]
    items = sorted(items, key=lambda w: w["sample_id"])
    order = rng.permutation(len(items))
    shuffled = [items[i] for i in order]

    n = len(shuffled)
    n_train = int(n * 0.60)
    n_val = int(n * 0.20)

    splits["train"].extend(shuffled[:n_train])
    splits["val"].extend(shuffled[n_train:n_train + n_val])
    splits["test"].extend(shuffled[n_train + n_val:])

# Shuffle each split for training.
for name in splits:
    rng.shuffle(splits[name])

def arrays(items):
    X = np.stack([w["x"] for w in items]).astype(np.float32)
    y = np.array([LABEL_TO_ID[w["label"]] for w in items], dtype=np.int32)
    return X, y

X_train, y_train = arrays(splits["train"])
X_val, y_val = arrays(splits["val"])
X_test, y_test = arrays(splits["test"])

print("\nSplit sizes:")
for name, items in splits.items():
    counts = {
        label: sum(w["label"] == label for w in items)
        for label in LABELS
    }
    print(f"  {name:5s}: {len(items)} windows; {counts}")

# Save exact split membership for reproducibility.
manifest = {
    name: [
        {"label": w["label"], "sample_id": w["sample_id"]}
        for w in items
    ]
    for name, items in splits.items()
}
with open(DATA_OUT / "split_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

# ---------------- Normalize: fit on training only ----------------
mean = X_train.reshape(-1, N_FEATURES).mean(axis=0)
std = X_train.reshape(-1, N_FEATURES).std(axis=0)
std[std < 1e-8] = 1.0

def normalize(X):
    return ((X - mean) / std).astype(np.float32)

X_train = normalize(X_train)
X_val = normalize(X_val)
X_test = normalize(X_test)

np.savez(DATA_OUT / "scaler_stats.npz", mean=mean, std=std)
with open(DATA_OUT / "label_map.json", "w") as f:
    json.dump(LABEL_TO_ID, f, indent=2)

np.savez_compressed(
    DATA_OUT / "splits_normalized.npz",
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
)

print("\nTraining-only scaler:")
print("  Mean:", np.round(mean, 5))
print("  Std: ", np.round(std, 5))

# ---------------- Model ----------------
def build_model():
    inputs = tf.keras.Input(
        shape=(WINDOW_SIZE, N_FEATURES), name="imu_input"
    )

    x = tf.keras.layers.Conv1D(
        32, 5, padding="same", activation="relu"
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv1D(
        64, 3, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv1D(
        64, 3, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(
        len(LABELS), activation="softmax", name="gesture_probabilities"
    )(x)

    return tf.keras.Model(inputs, outputs, name="edgeai_gesture_cnn")

model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()
print(f"Trainable + non-trainable parameters: {model.count_params():,}")

best_path = MODEL_OUT / "best_gesture_model.keras"
final_path = MODEL_OUT / "gesture_model.keras"

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
        min_lr=1e-5,
        verbose=1,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
]

print("\nStarting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2,
)

# EarlyStopping restores best weights; save that selected model.
model.save(final_path)

# ---------------- Evaluate once on held-out test set ----------------
probabilities = model.predict(X_test, verbose=0)
y_pred = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)
report = classification_report(
    y_test,
    y_pred,
    labels=list(range(len(LABELS))),
    target_names=LABELS,
    zero_division=0,
)
cm = confusion_matrix(
    y_test, y_pred, labels=list(range(len(LABELS)))
)

print("\n=== HELD-OUT TEST RESULTS ===")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro F1: {macro_f1:.4f}")
print("\nClassification report:")
print(report)
print("Confusion matrix (rows=true, columns=predicted):")
print("Labels:", LABELS)
print(cm)

with open(OUT / "evaluation_report.txt", "w") as f:
    f.write(f"TensorFlow: {tf.__version__}\n")
    f.write(f"Seed: {SEED}\n")
    f.write(f"Window count: {len(windows)}\n")
    f.write(f"Parameter count: {model.count_params()}\n")
    f.write(f"Test accuracy: {accuracy:.6f}\n")
    f.write(f"Test macro F1: {macro_f1:.6f}\n\n")
    f.write("Classification report:\n")
    f.write(report)
    f.write("\nConfusion matrix (rows=true, columns=predicted):\n")
    f.write(str(cm))
    f.write("\n\nLabel order:\n")
    f.write(json.dumps(LABEL_TO_ID, indent=2))

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history.history["loss"], label="Train")
axes[0].plot(history.history["val_loss"], label="Validation")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["accuracy"], label="Train")
axes[1].plot(history.history["val_accuracy"], label="Validation")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / "training_curves.png", dpi=140)
plt.close(fig)

print("\nSaved new artifacts under:", OUT)
print("  models/gesture_model.keras")
print("  models/best_gesture_model.keras")
print("  data/scaler_stats.npz")
print("  data/label_map.json")
print("  data/split_manifest.json")
print("  data/splits_normalized.npz")
print("  evaluation_report.txt")
print("  training_curves.png")
print("\nTraining stage complete. INT8 conversion has NOT been run yet.")
