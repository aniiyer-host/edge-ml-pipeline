import os
import numpy as np
import tensorflow as tf

# ==========================================================
# Experiment 5 - Task 2
# Post-Training Quantization
# ==========================================================

print("=" * 60)
print("EXPERIMENT 5 - TASK 2")
print("Post-Training Quantization")
print("=" * 60)

# ----------------------------------------------------------
# File Names
# ----------------------------------------------------------

MODEL_NAME = "TinyNN_Iris.keras"

OUTPUT_DIR = "tflite_models"

DYNAMIC_MODEL = os.path.join(
    OUTPUT_DIR,
    "TinyNN_dynamic.tflite"
)

FLOAT16_MODEL = os.path.join(
    OUTPUT_DIR,
    "TinyNN_float16.tflite"
)

INT8_MODEL = os.path.join(
    OUTPUT_DIR,
    "TinyNN_int8.tflite"
)

# ----------------------------------------------------------
# Create Output Directory
# ----------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------
# Check Required Files
# ----------------------------------------------------------

if not os.path.exists(MODEL_NAME):
    raise FileNotFoundError(
        f"'{MODEL_NAME}' was not found."
    )

if not os.path.exists("X_train.npy"):
    raise FileNotFoundError(
        "'X_train.npy' was not found."
    )

# ----------------------------------------------------------
# Load Model
# ----------------------------------------------------------

print("\nLoading baseline model...")

model = tf.keras.models.load_model(MODEL_NAME)

print("Model loaded successfully.")

# ----------------------------------------------------------
# Load Representative Dataset
# ----------------------------------------------------------

X_train = np.load("X_train.npy")

print(f"\nRepresentative dataset shape: {X_train.shape}")

# ==========================================================
# 1. DYNAMIC RANGE QUANTIZATION
# ==========================================================

print("\n" + "-" * 60)
print("1. Dynamic Range Quantization")
print("-" * 60)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

dynamic_tflite = converter.convert()

with open(DYNAMIC_MODEL, "wb") as f:
    f.write(dynamic_tflite)

print("Dynamic Range model created.")
print(f"Saved to: {DYNAMIC_MODEL}")

# ==========================================================
# 2. FLOAT16 QUANTIZATION
# ==========================================================

print("\n" + "-" * 60)
print("2. Float16 Quantization")
print("-" * 60)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.target_spec.supported_types = [
    tf.float16
]

float16_tflite = converter.convert()

with open(FLOAT16_MODEL, "wb") as f:
    f.write(float16_tflite)

print("Float16 model created.")
print(f"Saved to: {FLOAT16_MODEL}")

# ==========================================================
# 3. FULL INTEGER INT8 QUANTIZATION
# ==========================================================

print("\n" + "-" * 60)
print("3. Full Integer Int8 Quantization")
print("-" * 60)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

# ----------------------------------------------------------
# Representative Dataset Generator
# ----------------------------------------------------------

def representative_dataset():
    for sample in X_train:
        sample = sample.astype(np.float32)

        # Add batch dimension
        sample = np.expand_dims(sample, axis=0)

        yield [sample]


converter.representative_dataset = representative_dataset

# Force full integer quantization
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

int8_tflite = converter.convert()

with open(INT8_MODEL, "wb") as f:
    f.write(int8_tflite)

print("Full Integer Int8 model created.")
print(f"Saved to: {INT8_MODEL}")

# ==========================================================
# DISPLAY MODEL SIZES
# ==========================================================

dynamic_size = os.path.getsize(DYNAMIC_MODEL) / 1024
float16_size = os.path.getsize(FLOAT16_MODEL) / 1024
int8_size = os.path.getsize(INT8_MODEL) / 1024

print("\n" + "=" * 60)
print("QUANTIZATION RESULTS")
print("=" * 60)

print(f"Dynamic Range Model : {dynamic_size:.2f} KB")
print(f"Float16 Model       : {float16_size:.2f} KB")
print(f"Int8 Model          : {int8_size:.2f} KB")

print("=" * 60)
print("Task 2 Completed Successfully.")
print("=" * 60)

