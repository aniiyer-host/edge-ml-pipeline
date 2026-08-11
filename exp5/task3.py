import os
import time
import numpy as np

# ==========================================================
# IMPORTANT:
# Force TensorFlow/Keras APIs used by TFMOT to use tf_keras
# ==========================================================

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tf_keras
import tensorflow_model_optimization as tfmot

# Standalone Keras 3 is used ONLY to load the original
# Experiment 4 .keras model and extract its weights.
import keras


# ==========================================================
# Experiment 5 - Task 3
# Weight Pruning
# ==========================================================

print("=" * 60)
print("EXPERIMENT 5 - TASK 3")
print("Magnitude-Based Weight Pruning")
print("=" * 60)

# ----------------------------------------------------------
# File Names
# ----------------------------------------------------------

MODEL_NAME = "TinyNN_Iris.keras"

PRUNED_MODEL = "TinyNN_Iris_pruned.h5"

STRIPPED_MODEL = "TinyNN_Iris_pruned_stripped.h5"

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

print("\nLoading dataset...")

X_train = np.load("X_train.npy").astype(np.float32)
X_test = np.load("X_test.npy").astype(np.float32)

y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
print(f"Input shape      : {X_train.shape}")


# ==========================================================
# LOAD ORIGINAL KERAS 3 MODEL
# ==========================================================

print("\nLoading Experiment 4 Keras model...")

original_model = keras.models.load_model(
    MODEL_NAME,
    compile=False
)

print("Original model loaded successfully.")

print("\nOriginal Model")
print("-" * 60)

original_model.summary()

# ----------------------------------------------------------
# Get Dense Layers
# ----------------------------------------------------------

original_dense_layers = [
    layer
    for layer in original_model.layers
    if isinstance(layer, keras.layers.Dense)
]

if len(original_dense_layers) != 3:
    raise RuntimeError(
        f"Expected 3 Dense layers, "
        f"found {len(original_dense_layers)}."
    )

print(
    f"\nFound {len(original_dense_layers)} Dense layers."
)


# ==========================================================
# BASELINE ACCURACY
# ==========================================================

print("\nEvaluating baseline model...")

baseline_predictions = original_model.predict(
    X_test,
    verbose=0
)

baseline_predicted_labels = np.argmax(
    baseline_predictions,
    axis=1
)

baseline_accuracy = np.mean(
    baseline_predicted_labels == y_test
)

print(
    f"Baseline Accuracy : "
    f"{baseline_accuracy * 100:.2f}%"
)


# ==========================================================
# PRUNING CONFIGURATION
# ==========================================================

TARGET_SPARSITY = 0.70

BATCH_SIZE = 8
EPOCHS = 10

steps_per_epoch = int(
    np.ceil(len(X_train) / BATCH_SIZE)
)

END_STEP = steps_per_epoch * EPOCHS

print("\n" + "-" * 60)
print("Pruning Configuration")
print("-" * 60)

print(
    f"Target Sparsity     : "
    f"{TARGET_SPARSITY * 100:.0f}%"
)

print(f"Fine-Tuning Epochs : {EPOCHS}")
print(f"Batch Size         : {BATCH_SIZE}")
print(f"Pruning Steps      : {END_STEP}")


# ==========================================================
# PRUNING SCHEDULE
# ==========================================================

pruning_schedule = (
    tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=TARGET_SPARSITY,
        begin_step=0,
        end_step=END_STEP
    )
)


# ==========================================================
# CREATE LEGACY TF_KERAS MODEL
# ==========================================================

print("\nCreating tf_keras pruning model...")

inputs = tf_keras.Input(
    shape=(4,),
    name="input_layer"
)

# ----------------------------------------------------------
# First Dense Layer
# ----------------------------------------------------------

dense1 = tf_keras.layers.Dense(
    16,
    activation="relu",
    name="dense"
)

pruned_dense1 = (
    tfmot.sparsity.keras.prune_low_magnitude(
        dense1,
        pruning_schedule=pruning_schedule
    )
)

x = pruned_dense1(inputs)


# ----------------------------------------------------------
# Second Dense Layer
# ----------------------------------------------------------

dense2 = tf_keras.layers.Dense(
    8,
    activation="relu",
    name="dense_1"
)

pruned_dense2 = (
    tfmot.sparsity.keras.prune_low_magnitude(
        dense2,
        pruning_schedule=pruning_schedule
    )
)

x = pruned_dense2(x)


# ----------------------------------------------------------
# Output Dense Layer
# ----------------------------------------------------------

dense3 = tf_keras.layers.Dense(
    3,
    activation="softmax",
    name="dense_2"
)

pruned_dense3 = (
    tfmot.sparsity.keras.prune_low_magnitude(
        dense3,
        pruning_schedule=pruning_schedule
    )
)

outputs = pruned_dense3(x)


# ----------------------------------------------------------
# Create Model
# ----------------------------------------------------------

pruned_model = tf_keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="TinyNN_Iris_Pruned"
)

print("Pruned model created successfully.")


# ==========================================================
# COPY EXPERIMENT 4 WEIGHTS
# ==========================================================

print("\nCopying trained weights...")

pruned_dense_layers = [
    pruned_dense1.layer,
    pruned_dense2.layer,
    pruned_dense3.layer
]

for original_layer, pruned_layer in zip(
    original_dense_layers,
    pruned_dense_layers
):

    pruned_layer.set_weights(
        original_layer.get_weights()
    )

print("Trained weights copied successfully.")


# ==========================================================
# COMPILE PRUNED MODEL
# ==========================================================

pruned_model.compile(
    optimizer=tf_keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================================
# FINE-TUNING
# ==========================================================

print("\n" + "-" * 60)
print("Fine-Tuning Pruned Model")
print("-" * 60)

start_time = time.perf_counter()

history = pruned_model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        tfmot.sparsity.keras.UpdatePruningStep()
    ],
    verbose=1
)

training_time = (
    time.perf_counter() - start_time
)


# ==========================================================
# EVALUATE PRUNED MODEL
# ==========================================================

pruned_loss, pruned_accuracy = (
    pruned_model.evaluate(
        X_test,
        y_test,
        verbose=0
    )
)

print("\n" + "-" * 60)
print("Pruned Model Evaluation")
print("-" * 60)

print(
    f"Pruned Test Accuracy : "
    f"{pruned_accuracy * 100:.2f}%"
)

print(
    f"Pruned Test Loss     : "
    f"{pruned_loss:.4f}"
)

print(
    f"Fine-Tuning Time     : "
    f"{training_time:.2f} seconds"
)


# ==========================================================
# CALCULATE ACTUAL SPARSITY
# ==========================================================

print("\n" + "-" * 60)
print("Actual Model Sparsity")
print("-" * 60)

total_weights = 0
zero_weights = 0

for layer in pruned_dense_layers:

    kernel = layer.get_weights()[0]

    total_weights += kernel.size

    zero_weights += np.count_nonzero(
        kernel == 0
    )

actual_sparsity = (
    zero_weights / total_weights
) * 100

print(
    f"Total Kernel Weights : "
    f"{total_weights}"
)

print(
    f"Zero Weights         : "
    f"{zero_weights}"
)

print(
    f"Actual Sparsity      : "
    f"{actual_sparsity:.2f}%"
)


# ==========================================================
# SAVE PRUNED MODEL
# ==========================================================

print("\nSaving pruned model...")

pruned_model.save(
    PRUNED_MODEL
)

pruned_size = (
    os.path.getsize(PRUNED_MODEL) / 1024
)

print(
    f"Pruned Model Size : "
    f"{pruned_size:.2f} KB"
)


# ==========================================================
# STRIP PRUNING WRAPPERS
# ==========================================================

print("\nStripping pruning wrappers...")

stripped_model = (
    tfmot.sparsity.keras.strip_pruning(
        pruned_model
    )
)

stripped_model.save(
    STRIPPED_MODEL
)

stripped_size = (
    os.path.getsize(STRIPPED_MODEL) / 1024
)

print(
    f"Stripped Model Size : "
    f"{stripped_size:.2f} KB"
)


# ==========================================================
# VERIFY STRIPPED MODEL
# ==========================================================

print("\nVerifying stripped model...")

stripped_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

stripped_loss, stripped_accuracy = (
    stripped_model.evaluate(
        X_test,
        y_test,
        verbose=0
    )
)

print(
    f"Stripped Model Accuracy : "
    f"{stripped_accuracy * 100:.2f}%"
)


# ==========================================================
# FINAL RESULTS
# ==========================================================

print("\n" + "=" * 60)
print("TASK 3 RESULTS")
print("=" * 60)

print(
    f"Target Sparsity       : "
    f"{TARGET_SPARSITY * 100:.0f}%"
)

print(
    f"Actual Sparsity       : "
    f"{actual_sparsity:.2f}%"
)

print(
    f"Baseline Accuracy     : "
    f"{baseline_accuracy * 100:.2f}%"
)

print(
    f"Pruned Accuracy       : "
    f"{pruned_accuracy * 100:.2f}%"
)

print(
    f"Stripped Accuracy     : "
    f"{stripped_accuracy * 100:.2f}%"
)

print(
    f"Accuracy Change       : "
    f"{(pruned_accuracy - baseline_accuracy) * 100:+.2f}%"
)

print(
    f"Pruned Model Size     : "
    f"{pruned_size:.2f} KB"
)

print(
    f"Stripped Model Size   : "
    f"{stripped_size:.2f} KB"
)

print(
    f"Fine-Tuning Time      : "
    f"{training_time:.2f} seconds"
)

print("=" * 60)
print("Task 3 Completed Successfully.")
print("=" * 60)
