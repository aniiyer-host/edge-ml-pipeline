import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path("gesture-cmd/server/edgeai_v2")
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS / "gesture_model.keras"
SPLITS_PATH = DATA / "splits_normalized.npz"
LABEL_PATH = DATA / "label_map.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)
if not SPLITS_PATH.exists():
    raise FileNotFoundError(SPLITS_PATH)

print("TensorFlow:", tf.__version__)
print("Loading saved model:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

with np.load(SPLITS_PATH) as d:
    X_train = d["X_train"].astype(np.float32)
    y_train = d["y_train"]
    X_test = d["X_test"].astype(np.float32)
    y_test = d["y_test"]

with open(LABEL_PATH) as f:
    label_map = json.load(f)

# Keras reference predictions
keras_probs = model.predict(X_test, verbose=0)
keras_pred = np.argmax(keras_probs, axis=1)
keras_acc = accuracy_score(y_test, keras_pred)
keras_f1 = f1_score(y_test, keras_pred, average="macro", zero_division=0)

print(f"Keras test accuracy: {keras_acc:.4f}")
print(f"Keras test macro F1: {keras_f1:.4f}")

# Float32 TFLite
print("\nConverting float32 TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
float_bytes = converter.convert()

float_path = MODELS / "gesture_model_float32.tflite"
float_path.write_bytes(float_bytes)
print(f"Float32 model size: {len(float_bytes)} bytes")

def evaluate_tflite(model_bytes, X, y, integer_io):
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    predictions = []
    max_abs_error = 0.0

    for i, sample in enumerate(X):
        x = sample[None, ...].astype(np.float32)

        if integer_io:
            scale, zero_point = inp["quantization"]
            if scale <= 0:
                raise ValueError(f"Invalid input quantization scale: {scale}")

            # Round and clip before casting: never wrap values.
            q = np.rint(x / scale + zero_point)
            limits = np.iinfo(inp["dtype"])
            x = np.clip(q, limits.min, limits.max).astype(inp["dtype"])
        else:
            x = x.astype(inp["dtype"])

        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()
        output = interpreter.get_tensor(out["index"])

        if integer_io:
            scale, zero_point = out["quantization"]
            output = (output.astype(np.float32) - zero_point) * scale
        else:
            output = output.astype(np.float32)

        predictions.append(int(np.argmax(output[0])))

        if not integer_io:
            max_abs_error = max(
                max_abs_error,
                float(np.max(np.abs(output[0] - keras_probs[i])))
            )

    predictions = np.array(predictions)
    return (
        predictions,
        accuracy_score(y, predictions),
        f1_score(y, predictions, average="macro", zero_division=0),
        max_abs_error,
    )

float_pred, float_acc, float_f1, float_error = evaluate_tflite(
    float_bytes, X_test, y_test, integer_io=False
)
print(f"Float32 TFLite test accuracy: {float_acc:.4f}")
print(f"Float32 TFLite macro F1: {float_f1:.4f}")
print(f"Max Keras/TFLite output difference: {float_error:.8f}")

# Full integer quantization.
# Use all training windows as calibration examples (only 39, so cheap).
def representative_dataset():
    for sample in X_train:
        yield [sample[None, ...].astype(np.float32)]

print("\nConverting full INT8 TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

int8_bytes = converter.convert()
int8_path = MODELS / "gesture_model_int8.tflite"
int8_path.write_bytes(int8_bytes)
print(f"INT8 model size: {len(int8_bytes)} bytes")

int8_pred, int8_acc, int8_f1, _ = evaluate_tflite(
    int8_bytes, X_test, y_test, integer_io=True
)

print(f"INT8 TFLite test accuracy: {int8_acc:.4f}")
print(f"INT8 TFLite macro F1: {int8_f1:.4f}")
print(f"Float32 -> INT8 accuracy change: {(int8_acc-float_acc)*100:+.2f} percentage points")

# Inspect input/output quantization details for firmware implementation.
interpreter = tf.lite.Interpreter(model_content=int8_bytes)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

report = {
    "tensorflow_version": tf.__version__,
    "label_map": label_map,
    "keras_test_accuracy": float(keras_acc),
    "keras_test_macro_f1": float(keras_f1),
    "float32_tflite_test_accuracy": float(float_acc),
    "float32_tflite_macro_f1": float(float_f1),
    "keras_float32_max_abs_output_difference": float(float_error),
    "int8_tflite_test_accuracy": float(int8_acc),
    "int8_tflite_macro_f1": float(int8_f1),
    "float32_tflite_size_bytes": len(float_bytes),
    "int8_tflite_size_bytes": len(int8_bytes),
    "int8_input": {
        "shape": inp["shape"].tolist(),
        "dtype": str(inp["dtype"]),
        "scale": float(inp["quantization"][0]),
        "zero_point": int(inp["quantization"][1]),
    },
    "int8_output": {
        "shape": out["shape"].tolist(),
        "dtype": str(out["dtype"]),
        "scale": float(out["quantization"][0]),
        "zero_point": int(out["quantization"][1]),
    },
}

report_path = ROOT / "quantization_report.json"
report_path.write_text(json.dumps(report, indent=2))

print("\n=== QUANTIZATION REPORT ===")
print(json.dumps(report, indent=2))
print("\nSaved:", float_path)
print("Saved:", int8_path)
print("Saved:", report_path)
