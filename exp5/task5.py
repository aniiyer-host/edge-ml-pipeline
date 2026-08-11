import os
import time
import numpy as np
import tensorflow as tf

# ==========================================================
# Experiment 5 - Task 5
# Performance Comparison
# ==========================================================

print("=" * 70)
print("EXPERIMENT 5 - TASK 5")
print("Performance Comparison")
print("=" * 70)

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------

TFLITE_DIR = "tflite_models"

DYNAMIC_MODEL = os.path.join(
    TFLITE_DIR,
    "TinyNN_dynamic.tflite"
)

FLOAT16_MODEL = os.path.join(
    TFLITE_DIR,
    "TinyNN_float16.tflite"
)

INT8_MODEL = os.path.join(
    TFLITE_DIR,
    "TinyNN_int8.tflite"
)

PRUNED_MODEL = os.path.join(
    TFLITE_DIR,
    "TinyNN_pruned.tflite"
)

# ----------------------------------------------------------
# Load Test Dataset
# ----------------------------------------------------------

print("\nLoading test dataset...")

X_test = np.load(
    "X_test.npy"
).astype(np.float32)

y_test = np.load(
    "y_test.npy"
)

print(
    f"Test samples : {len(X_test)}"
)

# ==========================================================
# TFLITE EVALUATION FUNCTION
# ==========================================================

def evaluate_tflite(model_path):

    interpreter = tf.lite.Interpreter(
        model_path=model_path
    )

    interpreter.allocate_tensors()

    input_details = (
        interpreter.get_input_details()
    )

    output_details = (
        interpreter.get_output_details()
    )

    input_info = input_details[0]
    output_info = output_details[0]

    input_index = input_info["index"]
    output_index = output_info["index"]

    input_dtype = input_info["dtype"]
    output_dtype = output_info["dtype"]

    input_scale, input_zero_point = (
        input_info["quantization"]
    )

    output_scale, output_zero_point = (
        output_info["quantization"]
    )

    # ------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------

    sample = X_test[0:1]

    if input_dtype == np.int8:

        sample = (
            sample / input_scale
            + input_zero_point
        )

        sample = np.round(
            sample
        ).astype(np.int8)

    else:

        sample = sample.astype(
            input_dtype
        )

    interpreter.set_tensor(
        input_index,
        sample
    )

    interpreter.invoke()

    # ------------------------------------------------------
    # Inference Timing
    # ------------------------------------------------------

    predictions = []

    start = time.perf_counter()

    for sample in X_test:

        sample = np.expand_dims(
            sample,
            axis=0
        )

        # --------------------------------------------------
        # Quantized Input
        # --------------------------------------------------

        if input_dtype == np.int8:

            sample = (
                sample / input_scale
                + input_zero_point
            )

            sample = np.round(
                sample
            ).astype(np.int8)

        else:

            sample = sample.astype(
                input_dtype
            )

        # --------------------------------------------------
        # Invoke Model
        # --------------------------------------------------

        interpreter.set_tensor(
            input_index,
            sample
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_index
        )

        # --------------------------------------------------
        # Dequantize Output
        # --------------------------------------------------

        if output_dtype == np.int8:

            output = (
                output.astype(np.float32)
                - output_zero_point
            ) * output_scale

        predictions.append(
            output[0]
        )

    end = time.perf_counter()

    # ------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------

    predictions = np.array(
        predictions
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    accuracy = np.mean(
        predicted_labels == y_test
    )

    # ------------------------------------------------------
    # Inference Time
    # ------------------------------------------------------

    total_time_ms = (
        end - start
    ) * 1000

    average_time_ms = (
        total_time_ms
        / len(X_test)
    )

    # ------------------------------------------------------
    # File Size
    # ------------------------------------------------------

    model_size_kb = (
        os.path.getsize(model_path)
        / 1024
    )

    return (
        model_size_kb,
        accuracy,
        average_time_ms
    )


# ==========================================================
# EVALUATE MODELS
# ==========================================================

models = {
    "Dynamic Range": DYNAMIC_MODEL,
    "Float16": FLOAT16_MODEL,
    "Int8": INT8_MODEL,
    "Pruned": PRUNED_MODEL
}

results = {}

for name, path in models.items():

    print(
        f"\nEvaluating {name}..."
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    size, accuracy, inference = (
        evaluate_tflite(path)
    )

    results[name] = {
        "size": size,
        "accuracy": accuracy,
        "inference": inference
    }

    print(
        f"Size       : {size:.2f} KB"
    )

    print(
        f"Accuracy   : {accuracy * 100:.2f}%"
    )

    print(
        f"Inference  : {inference:.4f} ms/sample"
    )


# ==========================================================
# BASELINE INFORMATION
# ==========================================================

BASELINE_MODEL = "TinyNN_Iris.keras"

baseline_size = (
    os.path.getsize(BASELINE_MODEL)
    / 1024
)

baseline_model = tf.keras.models.load_model(
    BASELINE_MODEL,
    compile=False
)

baseline_predictions = (
    baseline_model.predict(
        X_test,
        verbose=0
    )
)

baseline_labels = np.argmax(
    baseline_predictions,
    axis=1
)

baseline_accuracy = np.mean(
    baseline_labels == y_test
)

# ----------------------------------------------------------
# Baseline inference time
# ----------------------------------------------------------

baseline_model.predict(
    X_test[:1],
    verbose=0
)

start = time.perf_counter()

baseline_model.predict(
    X_test,
    verbose=0
)

end = time.perf_counter()

baseline_inference = (
    (end - start)
    / len(X_test)
) * 1000


# ==========================================================
# FINAL COMPARISON TABLE
# ==========================================================

print("\n\n" + "=" * 90)
print("FINAL PERFORMANCE COMPARISON")
print("=" * 90)

print(
    f"{'Model':<22}"
    f"{'Size (KB)':<15}"
    f"{'Accuracy (%)':<18}"
    f"{'Inference (ms)':<18}"
)

print("-" * 90)

print(
    f"{'Baseline':<22}"
    f"{baseline_size:<15.2f}"
    f"{baseline_accuracy * 100:<18.2f}"
    f"{baseline_inference:<18.4f}"
)

for name, result in results.items():

    print(
        f"{name:<22}"
        f"{result['size']:<15.2f}"
        f"{result['accuracy'] * 100:<18.2f}"
        f"{result['inference']:<18.4f}"
    )

print("=" * 90)


# ==========================================================
# SIZE REDUCTION
# ==========================================================

print("\n" + "=" * 70)
print("MODEL SIZE REDUCTION")
print("=" * 70)

for name, result in results.items():

    reduction = (
        (baseline_size - result["size"])
        / baseline_size
    ) * 100

    print(
        f"{name:<20}: "
        f"{reduction:+.2f}%"
    )


# ==========================================================
# ACCURACY CHANGE
# ==========================================================

print("\n" + "=" * 70)
print("ACCURACY CHANGE")
print("=" * 70)

for name, result in results.items():

    change = (
        result["accuracy"]
        - baseline_accuracy
    ) * 100

    print(
        f"{name:<20}: "
        f"{change:+.2f} percentage points"
    )


# ==========================================================
# FINAL OBSERVATION
# ==========================================================

print("\n" + "=" * 70)
print("OBSERVATION")
print("=" * 70)

print(
    "The optimized models were compared with the "
    "original Tiny Neural Network using model size, "
    "test accuracy, and inference latency."
)

print(
    "The results demonstrate the trade-off between "
    "model compression, accuracy, and inference performance "
    "for Edge AI deployment."
)

print("=" * 70)
print("Task 5 Completed Successfully.")
print("=" * 70)
