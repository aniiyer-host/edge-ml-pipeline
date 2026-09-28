"""Task 5: profile accuracy, latency, and memory for the IMU gesture pipeline."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

try:
    from . import task4 as streaming
    from .task1 import HOP_SIZE, IMU0_COLUMNS, TARGET_SAMPLING_RATE_HZ, WINDOW_SIZE, resample_imu
    from .task2 import (
        FLOAT_MODEL_PATH,
        INT8_MODEL_PATH,
        PREPROCESSING_PATH,
        load_valid_recordings,
        split_recordings,
    )
    from .task3 import (
        NONE_CLASS_INDEX,
        load_runtime_preprocessing,
        score_recordings,
        select_validation_threshold,
        evaluate_recordings,
    )
except ImportError:
    import task4 as streaming
    from task1 import HOP_SIZE, IMU0_COLUMNS, TARGET_SAMPLING_RATE_HZ, WINDOW_SIZE, resample_imu
    from task2 import (
        FLOAT_MODEL_PATH,
        INT8_MODEL_PATH,
        PREPROCESSING_PATH,
        load_valid_recordings,
        split_recordings,
    )
    from task3 import (
        NONE_CLASS_INDEX,
        load_runtime_preprocessing,
        score_recordings,
        select_validation_threshold,
        evaluate_recordings,
    )

try:
    import resource
except ImportError:  # pragma: no cover - resource is unavailable on Windows.
    resource = None


OUTPUT_DIRECTORY = Path(__file__).resolve().parent
RESULTS_JSON_PATH = OUTPUT_DIRECTORY / "final_results.json"
CONFUSION_MATRIX_PATH = OUTPUT_DIRECTORY / "final_confusion_matrix.csv"
HOP_INTERVAL_MS = HOP_SIZE / TARGET_SAMPLING_RATE_HZ * 1000.0


def latency_summary(values: list[float] | np.ndarray) -> dict[str, float]:
    """Summarize a measured sequence of milliseconds."""
    samples = np.asarray(values, dtype=float)
    if samples.size == 0:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "mean_ms": float(np.mean(samples)),
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "max_ms": float(np.max(samples)),
    }


def process_peak_rss_mib() -> float | None:
    """Return whole-process peak RSS; this is not Tensor Arena memory."""
    if resource is None:
        return None
    peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and other common Unix platforms report KiB.
    if sys.platform == "darwin":
        return peak_rss / (1024.0 * 1024.0)
    return peak_rss / 1024.0


def theoretical_tensor_memory_bytes() -> dict[str, int]:
    """Estimate live host tensor buffers from their concrete NumPy dtypes."""
    samples = WINDOW_SIZE * len(IMU0_COLUMNS)
    ring_buffer = samples * np.dtype(np.float64).itemsize
    input_window = samples * np.dtype(np.float64).itemsize
    normalized_window = samples * np.dtype(np.float32).itemsize
    quantized_input = samples * np.dtype(np.int8).itemsize
    interpreter_input = samples * np.dtype(np.int8).itemsize
    output_tensor = 8 * np.dtype(np.int8).itemsize
    dequantized_probabilities = 8 * np.dtype(np.float32).itemsize
    named_buffers = {
        "ring_buffer_float64": ring_buffer,
        "input_window_float64": input_window,
        "normalized_window_float32": normalized_window,
        "quantized_input_int8": quantized_input,
        "tflite_input_tensor_int8": interpreter_input,
        "tflite_output_tensor_int8": output_tensor,
        "dequantized_probabilities_float32": dequantized_probabilities,
    }
    named_buffers["listed_host_tensor_working_set_bytes"] = sum(named_buffers.values())
    return named_buffers


def compute_trigger_responsiveness(
    test_scores: list,
    results_by_recording: list[list],
) -> dict[str, float | int | str | None]:
    """Estimate first correct trigger time from recording start, not gesture onset."""
    first_trigger_seconds = []
    window_seconds = WINDOW_SIZE / TARGET_SAMPLING_RATE_HZ
    hop_seconds = HOP_SIZE / TARGET_SAMPLING_RATE_HZ
    for recording, results in zip(test_scores, results_by_recording):
        if recording.true_class_index == NONE_CLASS_INDEX:
            continue
        for window_index, result in enumerate(results):
            if result.trigger_class_index == recording.true_class_index:
                first_trigger_seconds.append(window_seconds + window_index * hop_seconds)
                break

    return {
        "measured_correct_recordings": len(first_trigger_seconds),
        "mean_first_correct_trigger_seconds_from_recording_start": (
            float(np.mean(first_trigger_seconds)) if first_trigger_seconds else None
        ),
        "note": (
            "Recording-start-to-trigger proxy. Gesture-onset timestamps are not "
            "present in the label files, so onset responsiveness cannot be measured."
        ),
    }


def profile() -> dict:
    """Run final validation-threshold selection and one complete test profiling pass."""
    for required_path in (FLOAT_MODEL_PATH, INT8_MODEL_PATH, PREPROCESSING_PATH):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required Task 2/4 artifact not found: {required_path}")

    peak_rss_before_mib = process_peak_rss_mib()
    recordings, skipped = load_valid_recordings()
    splits = split_recordings(recordings)
    interpreter = tf.lite.Interpreter(model_path=str(INT8_MODEL_PATH), num_threads=1)
    interpreter.allocate_tensors()
    preprocessing = load_runtime_preprocessing(PREPROCESSING_PATH, interpreter)

    # Keep threshold selection on validation data only, matching Task 3.
    validation_scores = score_recordings(
        recordings, splits["validation"], interpreter, preprocessing
    )
    selected_threshold, _ = select_validation_threshold(validation_scores)

    test_indices = splits["test"]
    test_recordings = [recordings[int(index)] for index in test_indices]

    # Resampling is recording-level in this offline dataset simulation. Measure
    # it separately; Task 4's per-window total starts when the ring buffer emits
    # a complete window from the already-resampled 30 Hz stream.
    resampling_times_ms = []
    for recording in test_recordings:
        start = time.perf_counter()
        resample_imu(
            recording.timestamps,
            recording.imu_values,
            TARGET_SAMPLING_RATE_HZ,
        )
        resampling_times_ms.append((time.perf_counter() - start) * 1000.0)

    test_scores = score_recordings(recordings, test_indices, interpreter, preprocessing)
    trigger_metrics = evaluate_recordings(test_scores, selected_threshold)
    all_true_labels = np.concatenate(
        [
            np.full(score.probabilities.shape[0], score.true_class_index, dtype=np.int64)
            for score in test_scores
            if score.probabilities.shape[0]
        ]
    )
    all_probabilities = np.concatenate(
        [score.probabilities for score in test_scores if score.probabilities.shape[0]],
        axis=0,
    )
    raw_predictions = np.argmax(all_probabilities, axis=1)
    class_indices = np.arange(len(preprocessing.class_names))
    accuracy = float(accuracy_score(all_true_labels, raw_predictions))
    macro_f1 = float(
        f1_score(
            all_true_labels,
            raw_predictions,
            labels=class_indices,
            average="macro",
            zero_division=0,
        )
    )
    weighted_f1 = float(
        f1_score(
            all_true_labels,
            raw_predictions,
            labels=class_indices,
            average="weighted",
            zero_division=0,
        )
    )
    matrix = confusion_matrix(all_true_labels, raw_predictions, labels=class_indices)
    pd.DataFrame(
        matrix,
        index=preprocessing.class_names,
        columns=preprocessing.class_names,
    ).rename_axis("true_label").to_csv(CONFUSION_MATRIX_PATH)

    # Reuse the Task 4 streaming runner for actual sample-by-sample latency.
    # Disable only console event output; ring buffer, INT8 inference, and trigger
    # state are unchanged. Task 3 independently supplies window probabilities
    # for accuracy and is checked against Task 4's emitted trigger count.
    streaming.PRINT_WINDOW_EVENTS = False
    measured_latencies = []
    streaming_trigger_count = 0
    for session_number, recording in enumerate(test_recordings, start=1):
        session_latencies, session_triggers = streaming.run_streaming_session(
            recording,
            interpreter,
            preprocessing,
            selected_threshold,
            session_number,
            len(test_recordings),
        )
        measured_latencies.extend(session_latencies)
        streaming_trigger_count += session_triggers

    if not measured_latencies:
        raise ValueError("No test windows were available for profiling.")
    if len(measured_latencies) != all_true_labels.size:
        raise RuntimeError(
            "Task 4 stream window count does not match Task 3 test evaluation."
        )
    if streaming_trigger_count != trigger_metrics["trigger_count"]:
        raise RuntimeError(
            "Task 4 trigger count does not match Task 3 recording-level evaluation."
        )

    preprocessing_latency = latency_summary(
        [latency.preprocessing_ms for latency in measured_latencies]
    )
    resampling_latency = latency_summary(resampling_times_ms)
    inference_latency = latency_summary(
        [latency.inference_ms for latency in measured_latencies]
    )
    postprocessing_latency = latency_summary(
        [latency.postprocessing_ms for latency in measured_latencies]
    )
    total_latency = latency_summary(
        [latency.total_ms for latency in measured_latencies]
    )
    total_samples = np.asarray(
        [latency.total_ms for latency in measured_latencies], dtype=float
    )
    exceeded_count = int(np.count_nonzero(total_samples > HOP_INTERVAL_MS))
    exceeded_percentage = float(exceeded_count / total_samples.size * 100.0)

    responsiveness = compute_trigger_responsiveness(
        test_scores,
        trigger_metrics["results_by_recording"],
    )
    float_model = tf.keras.models.load_model(FLOAT_MODEL_PATH, compile=False)
    parameter_count = int(float_model.count_params())
    float_model_size_kb = FLOAT_MODEL_PATH.stat().st_size / 1024.0
    int8_model_size_kb = INT8_MODEL_PATH.stat().st_size / 1024.0
    size_reduction_percentage = (
        (float_model_size_kb - int8_model_size_kb) / float_model_size_kb * 100.0
    )
    memory_estimates = theoretical_tensor_memory_bytes()
    peak_rss_after_mib = process_peak_rss_mib()

    results = {
        "application": "IMU Gesture Recognition",
        "input": {
            "channels": list(IMU0_COLUMNS),
            "sampling_rate_hz": TARGET_SAMPLING_RATE_HZ,
            "window_length_seconds": WINDOW_SIZE / TARGET_SAMPLING_RATE_HZ,
            "window_size_samples": WINDOW_SIZE,
            "hop_stride_ms": HOP_INTERVAL_MS,
            "input_tensor_shape": [WINDOW_SIZE, len(IMU0_COLUMNS)],
        },
        "test_set": {
            "recordings": len(test_recordings),
            "windows": int(all_true_labels.size),
            "duration_hours": float(trigger_metrics["total_duration_hours"]),
            "skipped_invalid_recordings": len(skipped),
        },
        "model": {
            "parameter_count": parameter_count,
            "float_model_size_kb": float(float_model_size_kb),
            "int8_model_size_kb": float(int8_model_size_kb),
            "size_reduction_percentage": float(size_reduction_percentage),
            "int8_input_shape": [1, WINDOW_SIZE, len(IMU0_COLUMNS)],
            "int8_input_dtype": "int8",
            "int8_output_shape": [1, len(preprocessing.class_names)],
            "int8_output_dtype": "int8",
        },
        "classification": {
            "accuracy": accuracy,
            "accuracy_percent": accuracy * 100.0,
            "macro_f1": macro_f1,
            "macro_f1_percent": macro_f1 * 100.0,
            "weighted_f1": weighted_f1,
            "weighted_f1_percent": weighted_f1 * 100.0,
            "class_index_order": list(preprocessing.class_names),
            "confusion_matrix_csv": CONFUSION_MATRIX_PATH.name,
        },
        "latency": {
            "measurement_scope": (
                "Per completed window; total is Task 4 preprocessing + INT8 "
                "inference/dequantization + Task 3 post-processing."
            ),
            "timestamp_resampling_per_test_recording": resampling_latency,
            "preprocessing_per_window": preprocessing_latency,
            "int8_inference_per_window": inference_latency,
            "postprocessing_per_window": postprocessing_latency,
            "total_end_to_end_per_window": total_latency,
            "hop_budget_ms": HOP_INTERVAL_MS,
            "mean_real_time_margin_ms": HOP_INTERVAL_MS - total_latency["mean_ms"],
            "windows_exceeding_budget": exceeded_count,
            "windows_exceeding_budget_percentage": exceeded_percentage,
            "file_loading_and_console_output_included": False,
            "recording_level_resampling_included_in_per_window_total": False,
        },
        "trigger_performance": {
            "selected_threshold_from_validation": float(selected_threshold),
            "smoothing_window": streaming.SMOOTHING_WINDOW,
            "refractory_windows": streaming.REFRACTORY_WINDOWS,
            "total_triggers": int(trigger_metrics["trigger_count"]),
            "correct_triggers": int(trigger_metrics["correct_triggers"]),
            "false_triggers": int(trigger_metrics["false_triggers"]),
            "missed_gesture_triggers": int(
                trigger_metrics["missed_gesture_triggers"]
            ),
            "false_triggers_per_hour": float(
                trigger_metrics["false_triggers_per_hour"]
            ),
            "total_trigger_rate_per_hour": float(
                trigger_metrics["trigger_rate_per_hour"]
            ),
            "responsiveness": responsiveness,
            "correct_trigger_counting_rule": (
                "At most one matching trigger per labeled gesture recording; "
                "wrong-class, background, and repeated triggers count as false."
            ),
        },
        "memory": {
            "theoretical_host_tensor_memory_bytes": memory_estimates,
            "theoretical_host_tensor_memory_note": (
                "Host-side tensor storage estimate only; excludes Python objects, "
                "TFLite activations/workspaces, allocator overhead, and Tensor Arena SRAM."
            ),
            "process_peak_rss_mib": peak_rss_after_mib,
            "process_peak_rss_before_test_mib": peak_rss_before_mib,
            "process_peak_rss_note": (
                "Whole desktop Python process high-water RSS, including imported "
                "TensorFlow/runtime and model allocations; not incremental pipeline "
                "RAM and not Tensor Arena SRAM."
            ),
            "tensor_arena_sram": None,
            "tensor_arena_sram_note": (
                "Requires a TensorFlow Lite Micro deployment; this profiling run "
                "uses the desktop TensorFlow Lite interpreter."
            ),
        },
    }
    RESULTS_JSON_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print("\n=== Exp7 Final End-to-End Profiling ===")
    print("\nApplication: IMU Gesture Recognition")
    print(f"\nWindow length: {WINDOW_SIZE / TARGET_SAMPLING_RATE_HZ:.1f} s")
    print(f"Hop stride: {HOP_INTERVAL_MS:.0f} ms")
    print(f"Input tensor: ({WINDOW_SIZE}, {len(IMU0_COLUMNS)})")
    print(f"\nINT8 model size: {int8_model_size_kb:.2f} KB")
    print(f"Float model size: {float_model_size_kb:.2f} KB")
    print(f"Model size reduction: {size_reduction_percentage:.2f}%")
    print(f"\nParameters: {parameter_count}")
    print(f"\nTest accuracy: {accuracy:.2%}")
    print(f"Macro F1: {macro_f1:.2%}")
    print(f"Weighted F1: {weighted_f1:.2%}")
    print("\nPreprocessing latency (per completed window):")
    print(f"Mean: {preprocessing_latency['mean_ms']:.2f} ms")
    print(f"Median: {preprocessing_latency['median_ms']:.2f} ms")
    print(f"P95: {preprocessing_latency['p95_ms']:.2f} ms")
    print("\nTimestamp resampling latency (per test recording):")
    print(f"Mean: {resampling_latency['mean_ms']:.2f} ms")
    print(f"Median: {resampling_latency['median_ms']:.2f} ms")
    print(f"P95: {resampling_latency['p95_ms']:.2f} ms")
    print("\nINT8 inference latency:")
    print(f"Mean: {inference_latency['mean_ms']:.2f} ms")
    print(f"Median: {inference_latency['median_ms']:.2f} ms")
    print(f"P95: {inference_latency['p95_ms']:.2f} ms")
    print("\nPost-processing latency:")
    print(f"Mean: {postprocessing_latency['mean_ms']:.2f} ms")
    print(f"Median: {postprocessing_latency['median_ms']:.2f} ms")
    print(f"P95: {postprocessing_latency['p95_ms']:.2f} ms")
    print("\nTotal end-to-end latency:")
    print(f"Mean: {total_latency['mean_ms']:.2f} ms")
    print(f"Median: {total_latency['median_ms']:.2f} ms")
    print(f"P95: {total_latency['p95_ms']:.2f} ms")
    print(f"Maximum: {total_latency['max_ms']:.2f} ms")
    print("\n200 ms real-time budget:")
    print(f"Average latency: {total_latency['mean_ms']:.2f} ms")
    print(f"P95 latency: {total_latency['p95_ms']:.2f} ms")
    print(f"Exceeded: {exceeded_percentage:.2f}%")
    print(f"False triggers/hour: {trigger_metrics['false_triggers_per_hour']:.2f}")
    print("\nTrigger performance:")
    print(f"Total triggers: {trigger_metrics['trigger_count']}")
    print(f"Correct triggers: {trigger_metrics['correct_triggers']}")
    print(f"False triggers: {trigger_metrics['false_triggers']}")
    print(f"Missed triggers: {trigger_metrics['missed_gesture_triggers']}")
    response_seconds = responsiveness[
        "mean_first_correct_trigger_seconds_from_recording_start"
    ]
    if response_seconds is None:
        print("Mean first-correct-trigger time: unavailable")
    else:
        print(f"Mean first-correct-trigger time from recording start: {response_seconds:.2f} s")
    print("\nTheoretical host tensor memory estimates (not Tensor Arena SRAM):")
    for name, size in memory_estimates.items():
        if isinstance(size, int):
            print(f"{name}: {size} bytes")
    if peak_rss_after_mib is None:
        print("Process peak RSS: unavailable on this platform")
    else:
        print(f"Whole Python process peak RSS: {peak_rss_after_mib:.2f} MiB (not Tensor Arena SRAM)")
    print("Tensor Arena SRAM: not measured; requires TensorFlow Lite Micro deployment.")
    print(f"\nConfusion matrix saved: {CONFUSION_MATRIX_PATH.name}")
    print(f"Machine-readable results saved: {RESULTS_JSON_PATH.name}")

    return results


def main() -> None:
    profile()


if __name__ == "__main__":
    main()
