"""Task 3: smooth INT8 gesture predictions and evaluate trigger logic."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

try:
    from .task1 import (
        HOP_SIZE,
        IMU0_COLUMNS,
        TARGET_SAMPLING_RATE_HZ,
        WINDOW_SIZE,
        generate_windows,
        resample_imu,
    )
    from .task2 import (
        INT8_MODEL_PATH,
        LABEL_NAMES,
        LABEL_TO_INDEX,
        PREPROCESSING_PATH,
        SOURCE_LABELS,
        load_valid_recordings,
        split_recordings,
    )
except ImportError:
    from task1 import (
        HOP_SIZE,
        IMU0_COLUMNS,
        TARGET_SAMPLING_RATE_HZ,
        WINDOW_SIZE,
        generate_windows,
        resample_imu,
    )
    from task2 import (
        INT8_MODEL_PATH,
        LABEL_NAMES,
        LABEL_TO_INDEX,
        PREPROCESSING_PATH,
        SOURCE_LABELS,
        load_valid_recordings,
        split_recordings,
    )


SMOOTHING_WINDOW = 3
THRESHOLD_CANDIDATES = (0.50, 0.60, 0.70, 0.80, 0.90)
# Leave as None to select from validation; set a float to pin an operating point.
FINAL_THRESHOLD: float | None = None
REFRACTORY_WINDOWS = 3
RANDOM_SEED = 42
NONE_CLASS_INDEX = LABEL_TO_INDEX[10]


@dataclass(frozen=True)
class RuntimePreprocessing:
    """Training-derived normalization and model tensor quantization values."""

    means: np.ndarray
    standard_deviations: np.ndarray
    class_source_labels: tuple[int, ...]
    class_names: tuple[str, ...]
    input_scale: float
    input_zero_point: int
    output_scale: float
    output_zero_point: int


@dataclass(frozen=True)
class WindowResult:
    """Raw and post-processed output for one streaming window."""

    raw_class_index: int
    raw_confidence: float
    smoothed_class_index: int
    smoothed_confidence: float
    trigger_class_index: int | None


@dataclass
class RecordingScores:
    """Model probability vectors kept grouped by their source recording."""

    path: Path
    true_class_index: int
    duration_seconds: float
    probabilities: np.ndarray


class ConfidenceTrigger:
    """Moving-average confidence filter with one-shot hysteresis and refractory."""

    def __init__(
        self,
        smoothing_window: int,
        threshold: float,
        refractory_windows: int,
        none_class_index: int,
    ) -> None:
        if smoothing_window <= 0:
            raise ValueError("Smoothing window must be positive.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1.")
        if refractory_windows < 0:
            raise ValueError("Refractory period cannot be negative.")
        self.history: deque[np.ndarray] = deque(maxlen=smoothing_window)
        self.threshold = threshold
        self.refractory_windows = refractory_windows
        self.none_class_index = none_class_index
        self.active_class_index: int | None = None
        self.refractory_remaining = 0

    def update(self, probabilities: np.ndarray) -> WindowResult:
        """Smooth one probability vector, optionally trigger, and update state."""
        self.history.append(np.asarray(probabilities, dtype=np.float32).copy())
        smoothed = np.mean(np.stack(tuple(self.history)), axis=0)
        raw_class_index = int(np.argmax(probabilities))
        smoothed_class_index = int(np.argmax(smoothed))
        raw_confidence = float(probabilities[raw_class_index])
        smoothed_confidence = float(smoothed[smoothed_class_index])
        trigger_class_index = None

        in_refractory_period = self.refractory_remaining > 0
        if in_refractory_period:
            self.refractory_remaining -= 1

        is_confident_gesture = (
            smoothed_class_index != self.none_class_index
            and smoothed_confidence >= self.threshold
        )
        if self.active_class_index is not None and (
            not is_confident_gesture
            or smoothed_class_index != self.active_class_index
        ):
            # A confidence drop, NONE, or a different class rearms the detector.
            self.active_class_index = None

        if (
            self.active_class_index is None
            and not in_refractory_period
            and is_confident_gesture
        ):
            trigger_class_index = smoothed_class_index
            self.active_class_index = smoothed_class_index
            self.refractory_remaining = self.refractory_windows

        return WindowResult(
            raw_class_index=raw_class_index,
            raw_confidence=raw_confidence,
            smoothed_class_index=smoothed_class_index,
            smoothed_confidence=smoothed_confidence,
            trigger_class_index=trigger_class_index,
        )


def load_runtime_preprocessing(
    preprocessing_path: Path, interpreter: tf.lite.Interpreter
) -> RuntimePreprocessing:
    """Load Task 2 normalization/mapping and validate the INT8 model interface."""
    with np.load(preprocessing_path, allow_pickle=False) as saved:
        means = np.asarray(saved["channel_means"], dtype=np.float32)
        standard_deviations = np.asarray(
            saved["channel_standard_deviations"], dtype=np.float32
        )
        sampling_rate = int(saved["sampling_rate_hz"])
        window_size = int(saved["window_size"])
        hop_size = int(saved["hop_size"])
        saved_channel_names = tuple(str(name) for name in saved["channel_names"])
        saved_mapping = json.loads(str(saved["label_mapping"].item()))

    if means.shape != (len(IMU0_COLUMNS),) or standard_deviations.shape != means.shape:
        raise ValueError("Saved normalization parameters must contain six channels.")
    if not np.isfinite(means).all() or not np.isfinite(standard_deviations).all():
        raise ValueError("Saved normalization parameters contain non-finite values.")
    if np.any(standard_deviations <= 0):
        raise ValueError("Saved channel standard deviations must be positive.")
    if (sampling_rate, window_size, hop_size) != (
        TARGET_SAMPLING_RATE_HZ,
        WINDOW_SIZE,
        HOP_SIZE,
    ):
        raise ValueError("Saved sampling/window settings do not match Task 1.")
    if saved_channel_names != tuple(IMU0_COLUMNS):
        raise ValueError("Saved channel order does not match Task 1 IMU0 channels.")

    ordered_mapping = sorted(
        saved_mapping.items(), key=lambda item: int(item[1]["class_index"])
    )
    class_source_labels = tuple(int(source_label) for source_label, _ in ordered_mapping)
    class_names = tuple(str(value["name"]) for _, value in ordered_mapping)
    if class_source_labels != tuple(SOURCE_LABELS):
        raise ValueError("Saved model class order does not match Task 2's label mapping.")
    if class_names != tuple(LABEL_NAMES[label] for label in SOURCE_LABELS):
        raise ValueError("Saved class names do not match Task 2's label mapping.")

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    expected_input_shape = (1, WINDOW_SIZE, len(IMU0_COLUMNS))
    expected_output_shape = (1, len(class_source_labels))
    if tuple(int(value) for value in input_detail["shape"]) != expected_input_shape:
        raise ValueError(f"Unexpected TFLite input shape: {input_detail['shape']}")
    if tuple(int(value) for value in output_detail["shape"]) != expected_output_shape:
        raise ValueError(f"Unexpected TFLite output shape: {output_detail['shape']}")
    if input_detail["dtype"] is not np.int8 or output_detail["dtype"] is not np.int8:
        raise ValueError("Task 3 requires an INT8 input and output model.")

    input_scale, input_zero_point = input_detail["quantization"]
    output_scale, output_zero_point = output_detail["quantization"]
    if input_scale <= 0 or output_scale <= 0:
        raise ValueError("TFLite input and output quantization scales must be positive.")

    return RuntimePreprocessing(
        means=means,
        standard_deviations=standard_deviations,
        class_source_labels=class_source_labels,
        class_names=class_names,
        input_scale=float(input_scale),
        input_zero_point=int(input_zero_point),
        output_scale=float(output_scale),
        output_zero_point=int(output_zero_point),
    )


def infer_probabilities(
    interpreter: tf.lite.Interpreter,
    windows: np.ndarray,
    preprocessing: RuntimePreprocessing,
) -> np.ndarray:
    """Normalize, INT8-quantize, infer, and dequantize every window."""
    if windows.shape[0] == 0:
        return np.empty((0, len(preprocessing.class_source_labels)), dtype=np.float32)

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    normalized = (
        windows.astype(np.float32) - preprocessing.means
    ) / preprocessing.standard_deviations
    probabilities = np.empty(
        (normalized.shape[0], len(preprocessing.class_source_labels)), dtype=np.float32
    )

    for index, window in enumerate(normalized):
        quantized_input = np.rint(
            window / preprocessing.input_scale + preprocessing.input_zero_point
        )
        quantized_input = np.clip(
            quantized_input, np.iinfo(np.int8).min, np.iinfo(np.int8).max
        ).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], quantized_input[np.newaxis, ...])
        interpreter.invoke()
        quantized_output = interpreter.get_tensor(output_detail["index"])
        dequantized = (
            quantized_output.astype(np.float32) - preprocessing.output_zero_point
        ) * preprocessing.output_scale
        dequantized = np.maximum(dequantized[0], 0.0)
        probability_sum = float(dequantized.sum())
        if probability_sum <= 0:
            raise ValueError("TFLite output did not contain valid class probabilities.")
        probabilities[index] = dequantized / probability_sum

    return probabilities


def score_recordings(
    recordings: list,
    recording_indices: np.ndarray,
    interpreter: tf.lite.Interpreter,
    preprocessing: RuntimePreprocessing,
) -> list[RecordingScores]:
    """Resample and infer each split recording while preserving its boundary."""
    scored_recordings = []
    for recording_index in recording_indices:
        recording = recordings[int(recording_index)]
        duration = float(recording.timestamps[-1] - recording.timestamps[0])
        _, resampled_values = resample_imu(
            recording.timestamps,
            recording.imu_values,
            TARGET_SAMPLING_RATE_HZ,
        )
        windows = generate_windows(resampled_values, WINDOW_SIZE, HOP_SIZE)
        probabilities = infer_probabilities(interpreter, windows, preprocessing)
        true_class_index = LABEL_TO_INDEX[recording.source_label]
        scored_recordings.append(
            RecordingScores(
                path=recording.path,
                true_class_index=true_class_index,
                duration_seconds=duration,
                probabilities=probabilities,
            )
        )
    return scored_recordings


def process_recording(
    probabilities: np.ndarray,
    smoothing_window: int,
    threshold: float,
    refractory_windows: int,
) -> list[WindowResult]:
    """Apply fresh smoothing and trigger state to one complete recording."""
    detector = ConfidenceTrigger(
        smoothing_window=smoothing_window,
        threshold=threshold,
        refractory_windows=refractory_windows,
        none_class_index=NONE_CLASS_INDEX,
    )
    return [detector.update(probability) for probability in probabilities]


def evaluate_recordings(
    scored_recordings: list[RecordingScores], threshold: float
) -> dict[str, float | int | list[list[WindowResult]]]:
    """Measure window accuracy and recording-level trigger outcomes."""
    all_raw_predictions = []
    all_smoothed_predictions = []
    all_true_labels = []
    all_results = []
    trigger_count = 0
    correct_trigger_count = 0
    false_trigger_count = 0
    missed_gesture_count = 0

    for recording in scored_recordings:
        results = process_recording(
            recording.probabilities,
            smoothing_window=SMOOTHING_WINDOW,
            threshold=threshold,
            refractory_windows=REFRACTORY_WINDOWS,
        )
        all_results.append(results)
        if recording.probabilities.shape[0]:
            all_raw_predictions.extend(np.argmax(recording.probabilities, axis=1))
            all_smoothed_predictions.extend(
                result.smoothed_class_index for result in results
            )
            all_true_labels.extend(
                [recording.true_class_index] * len(results)
            )

        has_correct_trigger = False
        for result in results:
            if result.trigger_class_index is None:
                continue
            trigger_count += 1
            is_correct = (
                recording.true_class_index != NONE_CLASS_INDEX
                and result.trigger_class_index == recording.true_class_index
                and not has_correct_trigger
            )
            if is_correct:
                correct_trigger_count += 1
                has_correct_trigger = True
            else:
                # Only one trigger represents the one labeled gesture recording;
                # wrong-class and repeated triggers count as false triggers.
                false_trigger_count += 1

        if recording.true_class_index != NONE_CLASS_INDEX and not has_correct_trigger:
            missed_gesture_count += 1

    if not all_true_labels:
        raise ValueError("No test/validation windows were generated for evaluation.")

    true_labels = np.asarray(all_true_labels)
    raw_accuracy = float(np.mean(np.asarray(all_raw_predictions) == true_labels))
    postprocessed_accuracy = float(
        np.mean(np.asarray(all_smoothed_predictions) == true_labels)
    )
    total_duration_hours = sum(
        recording.duration_seconds for recording in scored_recordings
    ) / 3600.0
    if total_duration_hours <= 0:
        raise ValueError("Total evaluated recording duration must be positive.")
    false_triggers_per_hour = false_trigger_count / total_duration_hours
    trigger_rate_per_hour = trigger_count / total_duration_hours
    positive_recordings = sum(
        recording.true_class_index != NONE_CLASS_INDEX
        for recording in scored_recordings
    )
    false_negative_count = positive_recordings - correct_trigger_count
    f1_denominator = (
        2 * correct_trigger_count + false_trigger_count + false_negative_count
    )
    trigger_f1 = (
        0.0
        if f1_denominator == 0
        else (2 * correct_trigger_count) / f1_denominator
    )

    return {
        "raw_accuracy": raw_accuracy,
        "postprocessed_accuracy": postprocessed_accuracy,
        "trigger_count": trigger_count,
        "correct_triggers": correct_trigger_count,
        "false_triggers": false_trigger_count,
        "missed_gesture_triggers": missed_gesture_count,
        "false_triggers_per_hour": false_triggers_per_hour,
        "trigger_rate_per_hour": trigger_rate_per_hour,
        "total_duration_hours": total_duration_hours,
        "trigger_f1": trigger_f1,
        "results_by_recording": all_results,
    }


def select_validation_threshold(
    validation_scores: list[RecordingScores],
) -> tuple[float, list[tuple[float, dict]]]:
    """Choose the threshold with best validation recording-level trigger F1."""
    validation_results = [
        (threshold, evaluate_recordings(validation_scores, threshold))
        for threshold in THRESHOLD_CANDIDATES
    ]
    selected_threshold, _ = max(
        validation_results,
        key=lambda item: (
            item[1]["trigger_f1"],
            item[1]["correct_triggers"],
            -item[1]["false_triggers_per_hour"],
            -item[0],
        ),
    )
    if FINAL_THRESHOLD is not None:
        if not 0.0 <= FINAL_THRESHOLD <= 1.0:
            raise ValueError("FINAL_THRESHOLD must be between 0 and 1.")
        selected_threshold = FINAL_THRESHOLD
    return selected_threshold, validation_results


def print_threshold_table(validation_results: list[tuple[float, dict]]) -> None:
    """Print the validation-only confidence threshold comparison."""
    print("Threshold  Triggers  Correct  False  False/hour  Missed")
    for threshold, metrics in validation_results:
        print(
            f"{threshold:>9.2f}  {metrics['trigger_count']:>8}  "
            f"{metrics['correct_triggers']:>7}  {metrics['false_triggers']:>5}  "
            f"{metrics['false_triggers_per_hour']:>10.2f}  "
            f"{metrics['missed_gesture_triggers']:>6}"
        )


def print_prediction_trace(
    scored_recording: RecordingScores,
    results: list[WindowResult],
    class_source_labels: tuple[int, ...],
    class_names: tuple[str, ...],
    count: int = 5,
) -> None:
    """Show a short per-window raw/post-processed trace from one test recording."""
    print(f"\nPrediction trace: {scored_recording.path.name}")
    print("Window  Raw prediction        Smoothed prediction   Trigger")
    for index, result in enumerate(results[:count]):
        raw_label = class_source_labels[result.raw_class_index]
        smooth_label = class_source_labels[result.smoothed_class_index]
        trigger = (
            "NO"
            if result.trigger_class_index is None
            else f"YES: {class_names[result.trigger_class_index]}"
        )
        print(
            f"{index:>6}  {class_names[result.raw_class_index]} "
            f"({result.raw_confidence:.3f}, label {raw_label})  "
            f"{class_names[result.smoothed_class_index]} "
            f"({result.smoothed_confidence:.3f}, label {smooth_label})  {trigger}"
        )


def main() -> None:
    """Run validation threshold selection and one final test evaluation."""
    if not INT8_MODEL_PATH.is_file():
        raise FileNotFoundError(f"INT8 model not found: {INT8_MODEL_PATH}")
    if not PREPROCESSING_PATH.is_file():
        raise FileNotFoundError(f"Task 2 preprocessing file not found: {PREPROCESSING_PATH}")

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    recordings, skipped_recordings = load_valid_recordings()
    if skipped_recordings:
        print(f"Skipped invalid/unpaired recordings: {len(skipped_recordings)}")

    splits = split_recordings(recordings, random_seed=RANDOM_SEED)
    interpreter = tf.lite.Interpreter(model_path=str(INT8_MODEL_PATH))
    interpreter.allocate_tensors()
    preprocessing = load_runtime_preprocessing(PREPROCESSING_PATH, interpreter)

    validation_scores = score_recordings(
        recordings,
        splits["validation"],
        interpreter,
        preprocessing,
    )
    selected_threshold, validation_results = select_validation_threshold(
        validation_scores
    )

    test_scores = score_recordings(
        recordings,
        splits["test"],
        interpreter,
        preprocessing,
    )
    test_metrics = evaluate_recordings(test_scores, selected_threshold)

    print("=== Exp7 Task 3: Post-Processing Trigger Logic ===")
    print(f"Smoothing window: {SMOOTHING_WINDOW}")
    print(f"Selected confidence threshold: {selected_threshold:.2f}")
    print(f"Refractory period: {REFRACTORY_WINDOWS} windows")
    print("\nValidation:")
    print_threshold_table(validation_results)
    print(f"Selected threshold: {selected_threshold:.2f}")
    if FINAL_THRESHOLD is not None:
        print("Selected threshold was manually configured; the validation sweep is still shown.")

    print("\nTest:")
    print(f"Test recordings: {len(test_scores)}")
    print(f"Total test duration: {test_metrics['total_duration_hours']:.4f} hours")
    print(f"Raw prediction accuracy: {test_metrics['raw_accuracy']:.2%}")
    print(
        "Post-processed prediction accuracy: "
        f"{test_metrics['postprocessed_accuracy']:.2%}"
    )
    print(f"Total triggers: {test_metrics['trigger_count']}")
    print(f"Correct triggers: {test_metrics['correct_triggers']}")
    print(f"False triggers: {test_metrics['false_triggers']}")
    print(
        "Missed gesture triggers: "
        f"{test_metrics['missed_gesture_triggers']}"
    )
    print(f"Trigger rate: {test_metrics['trigger_rate_per_hour']:.2f} per hour")
    print(
        "False triggers/hour: "
        f"{test_metrics['false_triggers_per_hour']:.2f}"
    )
    print(
        "Trigger correctness is recording-level: at most one matching trigger per "
        "gesture recording is counted correct; repeats and wrong/background triggers are false."
    )

    results_by_recording = test_metrics["results_by_recording"]
    for index, recording in enumerate(test_scores):
        if recording.probabilities.shape[0]:
            print_prediction_trace(
                recording,
                results_by_recording[index],
                preprocessing.class_source_labels,
                preprocessing.class_names,
            )
            break


if __name__ == "__main__":
    main()
