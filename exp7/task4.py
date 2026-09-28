"""Task 4: stream recorded IMU samples through the INT8 gesture pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

try:
    from .task1 import (
        HOP_SIZE,
        IMU0_COLUMNS,
        TARGET_SAMPLING_RATE_HZ,
        WINDOW_SIZE,
        RingBuffer,
        resample_imu,
    )
    from .task2 import (
        INT8_MODEL_PATH,
        PREPROCESSING_PATH,
        Recording,
        load_valid_recordings,
        split_recordings,
    )
    from .task3 import (
        REFRACTORY_WINDOWS,
        SMOOTHING_WINDOW,
        ConfidenceTrigger,
        RuntimePreprocessing,
        score_recordings,
        select_validation_threshold,
        load_runtime_preprocessing,
    )
except ImportError:
    from task1 import (
        HOP_SIZE,
        IMU0_COLUMNS,
        TARGET_SAMPLING_RATE_HZ,
        WINDOW_SIZE,
        RingBuffer,
        resample_imu,
    )
    from task2 import (
        INT8_MODEL_PATH,
        PREPROCESSING_PATH,
        Recording,
        load_valid_recordings,
        split_recordings,
    )
    from task3 import (
        REFRACTORY_WINDOWS,
        SMOOTHING_WINDOW,
        ConfidenceTrigger,
        RuntimePreprocessing,
        score_recordings,
        select_validation_threshold,
        load_runtime_preprocessing,
    )


HOP_INTERVAL_MS = HOP_SIZE / TARGET_SAMPLING_RATE_HZ * 1000.0
PRINT_WINDOW_EVENTS = True


@dataclass(frozen=True)
class WindowLatency:
    """Measured compute time for a single completed sliding window."""

    preprocessing_ms: float
    inference_ms: float
    postprocessing_ms: float
    total_ms: float


def run_streaming_session(
    recording: Recording,
    interpreter: tf.lite.Interpreter,
    preprocessing: RuntimePreprocessing,
    threshold: float,
    session_number: int,
    session_count: int,
) -> tuple[list[WindowLatency], int]:
    """Replay one recording sample-by-sample with fresh ring and trigger state."""
    # CSV loading and timestamp resampling happen before the measured stream.
    _, resampled_stream = resample_imu(
        recording.timestamps,
        recording.imu_values,
        TARGET_SAMPLING_RATE_HZ,
    )
    ring_buffer = RingBuffer(capacity=WINDOW_SIZE, hop_size=HOP_SIZE)
    trigger = ConfidenceTrigger(
        smoothing_window=SMOOTHING_WINDOW,
        threshold=threshold,
        refractory_windows=REFRACTORY_WINDOWS,
        none_class_index=preprocessing.class_source_labels.index(10),
    )
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    latencies = []
    trigger_count = 0

    print(
        f"\n--- Streaming session {session_number}/{session_count}: "
        f"{recording.path.name} | ground truth: "
        f"{preprocessing.class_names[preprocessing.class_source_labels.index(recording.source_label)]} ---"
    )

    # Replay without sleeping: measured compute latency can be compared directly
    # with the 200 ms hop budget without making this offline test take minutes.
    for sample_index, sample in enumerate(resampled_stream):
        window_start = time.perf_counter()
        window = ring_buffer.append(sample)
        if window is None:
            continue
        if window.shape != (WINDOW_SIZE, len(IMU0_COLUMNS)):
            raise RuntimeError(f"Unexpected generated window shape: {window.shape}")

        normalized = (
            window.astype(np.float32) - preprocessing.means
        ) / preprocessing.standard_deviations
        quantized_input = np.rint(
            normalized / preprocessing.input_scale + preprocessing.input_zero_point
        )
        quantized_input = np.clip(
            quantized_input, np.iinfo(np.int8).min, np.iinfo(np.int8).max
        ).astype(np.int8)
        inference_start = time.perf_counter()
        preprocessing_ms = (inference_start - window_start) * 1000.0

        interpreter.set_tensor(input_detail["index"], quantized_input[np.newaxis, ...])
        interpreter.invoke()
        quantized_output = interpreter.get_tensor(output_detail["index"])
        output_probabilities = (
            quantized_output.astype(np.float32) - preprocessing.output_zero_point
        ) * preprocessing.output_scale
        output_probabilities = np.maximum(output_probabilities[0], 0.0)
        probability_sum = float(output_probabilities.sum())
        if probability_sum <= 0:
            raise ValueError("TFLite output did not contain valid class probabilities.")
        probabilities = output_probabilities / probability_sum
        postprocessing_start = time.perf_counter()
        inference_ms = (postprocessing_start - inference_start) * 1000.0

        result = trigger.update(probabilities)
        window_end = time.perf_counter()
        postprocessing_ms = (window_end - postprocessing_start) * 1000.0
        total_ms = (window_end - window_start) * 1000.0
        latencies.append(
            WindowLatency(
                preprocessing_ms=preprocessing_ms,
                inference_ms=inference_ms,
                postprocessing_ms=postprocessing_ms,
                total_ms=total_ms,
            )
        )

        if result.trigger_class_index is not None:
            trigger_count += 1
        if PRINT_WINDOW_EVENTS:
            predicted_name = preprocessing.class_names[result.smoothed_class_index]
            trigger_name = (
                "NONE"
                if result.trigger_class_index is None
                else preprocessing.class_names[result.trigger_class_index]
            )
            stream_time_seconds = (sample_index + 1) / TARGET_SAMPLING_RATE_HZ
            print(f"[time={stream_time_seconds:.2f}s]")
            print(
                f"Prediction: {predicted_name} "
                f"(raw={preprocessing.class_names[result.raw_class_index]})"
            )
            print(f"Confidence: {result.smoothed_confidence:.2f}")
            print(f"Trigger: {trigger_name}")

    return latencies, trigger_count


def main() -> None:
    """Calibrate from validation recordings, then replay each test recording."""
    if not INT8_MODEL_PATH.is_file():
        raise FileNotFoundError(f"INT8 model not found: {INT8_MODEL_PATH}")
    if not PREPROCESSING_PATH.is_file():
        raise FileNotFoundError(f"Preprocessing archive not found: {PREPROCESSING_PATH}")

    recordings, skipped_recordings = load_valid_recordings()
    splits = split_recordings(recordings)
    interpreter = tf.lite.Interpreter(model_path=str(INT8_MODEL_PATH), num_threads=1)
    interpreter.allocate_tensors()
    preprocessing = load_runtime_preprocessing(PREPROCESSING_PATH, interpreter)

    validation_scores = score_recordings(
        recordings,
        splits["validation"],
        interpreter,
        preprocessing,
    )
    selected_threshold, _ = select_validation_threshold(validation_scores)
    print(f"Validation-selected confidence threshold: {selected_threshold:.2f}")
    if skipped_recordings:
        print(f"Skipped invalid/unpaired recordings: {len(skipped_recordings)}")

    all_latencies = []
    total_triggers = 0
    test_indices = splits["test"]
    for session_number, recording_index in enumerate(test_indices, start=1):
        recording = recordings[int(recording_index)]
        session_latencies, session_triggers = run_streaming_session(
            recording,
            interpreter,
            preprocessing,
            selected_threshold,
            session_number,
            len(test_indices),
        )
        all_latencies.extend(session_latencies)
        total_triggers += session_triggers

    if not all_latencies:
        raise ValueError("No complete test windows were generated.")

    preprocessing_times = np.asarray(
        [latency.preprocessing_ms for latency in all_latencies], dtype=float
    )
    inference_times = np.asarray(
        [latency.inference_ms for latency in all_latencies], dtype=float
    )
    postprocessing_times = np.asarray(
        [latency.postprocessing_ms for latency in all_latencies], dtype=float
    )
    total_times = np.asarray([latency.total_ms for latency in all_latencies], dtype=float)
    exceeded_budget = int(np.count_nonzero(total_times > HOP_INTERVAL_MS))
    exceeded_percentage = exceeded_budget / total_times.size * 100.0
    average_total_ms = float(total_times.mean())
    real_time_margin_ms = HOP_INTERVAL_MS - average_total_ms

    print("\n=== Exp7 Task 4: End-to-End Real-Time Execution ===")
    print(f"Stream rate: {TARGET_SAMPLING_RATE_HZ} Hz")
    print(f"Window: {WINDOW_SIZE / TARGET_SAMPLING_RATE_HZ:.1f} s")
    print(f"Hop: {HOP_INTERVAL_MS:.0f} ms")
    print(f"Validation-selected threshold: {selected_threshold:.2f}")
    print(f"Test recordings streamed: {len(test_indices)}")
    print(f"Windows processed: {total_times.size}")
    print("\nAverage preprocessing latency: " f"{preprocessing_times.mean():.2f} ms")
    print(f"Average inference latency: {inference_times.mean():.2f} ms")
    print(f"Average post-processing latency: {postprocessing_times.mean():.2f} ms")
    print(f"Average total latency: {average_total_ms:.2f} ms")
    print(f"\nMedian total latency: {np.median(total_times):.2f} ms")
    print(f"95th percentile: {np.percentile(total_times, 95):.2f} ms")
    print(f"Maximum latency: {total_times.max():.2f} ms")
    print(f"\nHop budget: {HOP_INTERVAL_MS:.0f} ms")
    print(f"Real-time margin: {real_time_margin_ms:.2f} ms")
    print(
        f"Windows exceeding budget: {exceeded_budget} "
        f"({exceeded_percentage:.2f}%)"
    )
    print(f"\nTriggers generated: {total_triggers}")
    print(
        "Latency excludes CSV loading, recording-level resampling, and console output; "
        "the stream is replayed as fast as the CPU can process it."
    )


if __name__ == "__main__":
    main()
