import os
import time
import numpy as np
import tensorflow as tf

# ==========================================================
# Experiment 5 - Task 4
# Model Conversion and Evaluation
# ==========================================================

print("=" * 60)
print("EXPERIMENT 5 - TASK 4")
print("Model Conversion and Evaluation")
print("=" * 60)

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

PRUNED_KERAS_MODEL = (
    "TinyNN_Iris_pruned_stripped.h5"
)

PRUNED_TFLITE_MODEL = os.path.join(
    TFLITE_DIR,
    "TinyNN_pruned.tflite"
)

# ----------------------------------------------------------
# Load Test Dataset
# ----------------------------------------------------------

print("\nLoading test dataset...")

X_test = np.load("X_test.npy").astype(np.float32)
y_test = np.load("y_test.npy")

print(f"Test samples : {len(X_test)}")
print(f"Input shape  : {X_test.shape}")

# ----------------------------------------------------------
# Check Required TFLite Models
# ----------------------------------------------------------

required_models = [
    DYNAMIC_MODEL,
    FLOAT16_MODEL,
    INT8_MODEL
]

for model_path in required_models:

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Required model not found: {model_path}"
        )

# ----------------------------------------------------------
# Create Output Directory
# ----------------------------------------------------------

os.makedirs(TFLITE_DIR, exist_ok=True)


# ==========================================================
# CONVERT PRUNED MODEL TO TFLITE
# ==========================================================

print("\n" + "-" * 60)
print("Converting Pruned Model to TFLite")
print("-" * 60)

if not os.path.exists(PRUNED_KERAS_MODEL):

    raise FileNotFoundError(
        f"Pruned model not found: "
        f"{PRUNED_KERAS_MODEL}"
    )

pruned_model = tf.keras.models.load_model(
    PRUNED_KERAS_MODEL,
    compile=False
)

converter = tf.lite.TFLiteConverter.from_keras_model(
    pruned_model
)

pruned_tflite = converter.convert()

with open(PRUNED_TFLITE_MODEL, "wb") as f:
    f.write(pruned_tflite)

print(
    f"Pruned TFLite model saved to:\n"
    f"{PRUNED_TFLITE_MODEL}"
)


# ==========================================================
# HELPER FUNCTION
# ==========================================================

def get_interpreter(model_path):
    """
    Create a TFLite interpreter.

    Uses the standard tf.lite.Interpreter API
    supported by TensorFlow 2.21.
    """

    interpreter = tf.lite.Interpreter(
        model_path=model_path
    )

    interpreter.allocate_tensors()

    return interpreter


# ==========================================================
# EVALUATE TFLITE MODEL
# ==========================================================

def evaluate_tflite_model(
    model_path,
    X_test,
    y_test
):

    interpreter = get_interpreter(
        model_path
    )

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

    predictions = []

    # ------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------

    first_sample = X_test[0:1]

    if input_dtype == np.int8:

        first_sample = (
            first_sample / input_scale
            + input_zero_point
        ).round().astype(np.int8)

    else:

        first_sample = first_sample.astype(
            input_dtype
        )

    interpreter.set_tensor(
        input_index,
        first_sample
    )

    interpreter.invoke()

    # ------------------------------------------------------
    # Measure Inference
    # ------------------------------------------------------

    start_time = time.perf_counter()

    for sample in X_test:

        sample = np.expand_dims(
            sample,
            axis=0
        )

        # --------------------------------------------------
        # Handle Quantized Input
        # --------------------------------------------------

        if input_dtype == np.int8:

            if input_scale == 0:

                raise ValueError(
                    "Invalid input quantization scale."
                )

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
        # Inference
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
        # Handle Quantized Output
        # --------------------------------------------------

        if output_dtype == np.int8:

            output = (
                output.astype(np.float32)
                - output_zero_point
            ) * output_scale

        predictions.append(
            output[0]
        )

    end_time = time.perf_counter()

    # ------------------------------------------------------
    # Calculate Accuracy
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
    # Calculate Timing
    # ------------------------------------------------------

    total_time = (
        end_time - start_time
    )

    average_time_ms = (
        total_time / len(X_test)
    ) * 1000

    # ------------------------------------------------------
    # Model Size
    # ------------------------------------------------------

    model_size_kb = (
        os.path.getsize(model_path)
        / 1024
    )

    return (
        accuracy,
        average_time_ms,
        model_size_kb,
        input_dtype,
        output_dtype
    )


# ==========================================================
# EVALUATE ALL MODELS
# ==========================================================

models = {
    "Dynamic Range": DYNAMIC_MODEL,
    "Float16": FLOAT16_MODEL,
    "Int8": INT8_MODEL,
    "Pruned": PRUNED_TFLITE_MODEL
}

results = {}


for model_name, model_path in models.items():

    print("\n" + "-" * 60)
    print(f"Evaluating {model_name} Model")
    print("-" * 60)

    (
        accuracy,
        inference_time,
        model_size,
        input_dtype,
        output_dtype
    ) = evaluate_tflite_model(
        model_path,
        X_test,
        y_test
    )

    results[model_name] = {
        "size": model_size,
        "accuracy": accuracy,
        "inference_time": inference_time,
        "input_dtype": input_dtype,
        "output_dtype": output_dtype
    }

    print(
        f"Model Size       : "
        f"{model_size:.2f} KB"
    )

    print(
        f"Test Accuracy    : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Inference Time   : "
        f"{inference_time:.4f} ms/sample"
    )

    print(
        f"Input Data Type  : "
        f"{input_dtype}"
    )

    print(
        f"Output Data Type : "
        f"{output_dtype}"
    )


# ==========================================================
# FINAL TASK 4 RESULTS
# ==========================================================

print("\n" + "=" * 70)
print("TASK 4 - MODEL CONVERSION AND EVALUATION RESULTS")
print("=" * 70)

print(
    f"{'Model':<18}"
    f"{'Size (KB)':<15}"
    f"{'Accuracy (%)':<17}"
    f"{'Inference (ms)':<18}"
)

print("-" * 70)

for model_name, result in results.items():

    print(
        f"{model_name:<18}"
        f"{result['size']:<15.2f}"
        f"{result['accuracy'] * 100:<17.2f}"
        f"{result['inference_time']:<18.4f}"
    )

print("=" * 70)

print("\nGenerated TFLite Models:")

for model_name, model_path in models.items():

    print(
        f"{model_name:<18}: "
        f"{model_path}"
    )

print("\nTask 4 Completed Successfully.")
