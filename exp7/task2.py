"""Task 2: train and fully quantize a compact IMU gesture classifier."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

try:
    from .task1 import (
        HOP_SIZE,
        IMU0_COLUMNS,
        TARGET_SAMPLING_RATE_HZ,
        WINDOW_SIZE,
        generate_windows,
        load_recording,
        resample_imu,
    )
except ImportError:
    from task1 import (
        HOP_SIZE,
        IMU0_COLUMNS,
        TARGET_SAMPLING_RATE_HZ,
        WINDOW_SIZE,
        generate_windows,
        load_recording,
        resample_imu,
    )


RANDOM_SEED = 42
MAX_EPOCHS = 50
BATCH_SIZE = 64
REPRESENTATIVE_SAMPLE_COUNT = 200
DATA_DIRECTORY = Path(__file__).resolve().parent / "rosbag" / "data" / "data"
LABEL_DIRECTORY = Path(__file__).resolve().parent / "rosbag" / "data" / "label"
FLOAT_MODEL_PATH = Path(__file__).resolve().parent / "gesture_cnn_float.keras"
INT8_MODEL_PATH = Path(__file__).resolve().parent / "gesture_cnn_int8.tflite"
PREPROCESSING_PATH = Path(__file__).resolve().parent / "preprocessing.npz"

# Model outputs use contiguous indices; the source label 10 becomes class index 7.
LABEL_NAMES = {
    0: "STATIC",
    1: "SLIDE_UP",
    2: "SLIDE_DOWN",
    3: "SLIDE_LEFT",
    4: "SLIDE_RIGHT",
    5: "RELEASE",
    6: "GRASP",
    10: "NONE",
}
SOURCE_LABELS = tuple(LABEL_NAMES)
LABEL_TO_INDEX = {label: index for index, label in enumerate(SOURCE_LABELS)}
CLASS_NAMES = [LABEL_NAMES[label] for label in SOURCE_LABELS]


@dataclass
class Recording:
    """Validated source recording kept intact until recording-level splitting."""

    path: Path
    source_label: int
    timestamps: np.ndarray
    imu_values: np.ndarray


def load_recording_label(label_path: Path) -> int:
    """Read the one gesture label stored in a paired label CSV."""
    frame = pd.read_csv(label_path)
    if "label" not in frame.columns:
        raise ValueError("label CSV must contain a 'label' column")

    values = pd.to_numeric(frame["label"], errors="coerce").dropna().unique()
    if values.size != 1:
        raise ValueError("label CSV must contain exactly one unique non-empty label")
    numeric_label = float(values[0])
    if not numeric_label.is_integer():
        raise ValueError(f"label must be an integer, got {numeric_label}")
    source_label = int(numeric_label)
    if source_label not in LABEL_TO_INDEX:
        raise ValueError(f"unrecognized gesture label: {source_label}")
    return source_label


def load_valid_recordings(
    data_directory: Path = DATA_DIRECTORY,
    label_directory: Path = LABEL_DIRECTORY,
) -> tuple[list[Recording], list[tuple[str, str]]]:
    """Load valid data/label pairs and record any pairs that cannot be used."""
    if not data_directory.is_dir() or not label_directory.is_dir():
        raise FileNotFoundError(
            f"Expected recording folders at {data_directory} and {label_directory}"
        )

    recordings = []
    skipped = []
    data_paths = sorted(data_directory.glob("*_data.csv"))
    if not data_paths:
        raise FileNotFoundError(f"No *_data.csv recordings found in {data_directory}")

    for data_path in data_paths:
        label_path = label_directory / data_path.name.replace("_data.csv", "_label.csv")
        try:
            if not label_path.is_file():
                raise FileNotFoundError(f"paired label file not found: {label_path.name}")
            source_label = load_recording_label(label_path)
            timestamps, imu_values = load_recording(data_path)
            recordings.append(
                Recording(
                    path=data_path,
                    source_label=source_label,
                    timestamps=timestamps,
                    imu_values=imu_values,
                )
            )
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            skipped.append((data_path.name, str(error)))

    if not recordings:
        raise ValueError("No valid paired recordings were found.")
    return recordings, skipped


def split_recordings(
    recordings: list[Recording], random_seed: int = RANDOM_SEED
) -> dict[str, np.ndarray]:
    """Split recording indices first, with stratification across gesture labels."""
    indices = np.arange(len(recordings))
    labels = np.asarray([recording.source_label for recording in recordings])
    try:
        train_indices, remainder_indices = train_test_split(
            indices,
            test_size=0.30,
            random_state=random_seed,
            stratify=labels,
        )
        validation_indices, test_indices = train_test_split(
            remainder_indices,
            test_size=0.50,
            random_state=random_seed,
            stratify=labels[remainder_indices],
        )
    except ValueError as error:
        print(f"Warning: recording-level stratification unavailable ({error}); using seeded random splits.")
        train_indices, remainder_indices = train_test_split(
            indices, test_size=0.30, random_state=random_seed
        )
        validation_indices, test_indices = train_test_split(
            remainder_indices, test_size=0.50, random_state=random_seed
        )

    return {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
    }


def make_window_dataset(
    recordings: list[Recording], recording_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Resample recordings and label every complete window from its source."""
    windows_by_recording = []
    labels_by_recording = []
    short_recordings = []

    for recording_index in recording_indices:
        recording = recordings[int(recording_index)]
        _, resampled_values = resample_imu(
            recording.timestamps,
            recording.imu_values,
            TARGET_SAMPLING_RATE_HZ,
        )
        windows = generate_windows(resampled_values, WINDOW_SIZE, HOP_SIZE)
        if windows.shape[0] == 0:
            short_recordings.append(recording.path.name)
            continue

        windows_by_recording.append(windows.astype(np.float32))
        encoded_label = LABEL_TO_INDEX[recording.source_label]
        labels_by_recording.append(
            np.full(windows.shape[0], encoded_label, dtype=np.int64)
        )

    if not windows_by_recording:
        return (
            np.empty((0, WINDOW_SIZE, len(IMU0_COLUMNS)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            short_recordings,
        )
    return (
        np.concatenate(windows_by_recording, axis=0),
        np.concatenate(labels_by_recording, axis=0),
        short_recordings,
    )


def fit_channel_normalization(training_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-channel standardization using training windows only."""
    if training_windows.size == 0:
        raise ValueError("Cannot calculate normalization from an empty training set.")
    training_samples = training_windows.reshape(-1, training_windows.shape[-1]).astype(
        np.float64
    )
    means = training_samples.mean(axis=0)
    standard_deviations = training_samples.std(axis=0)
    standard_deviations[standard_deviations == 0] = 1.0
    return means.astype(np.float32), standard_deviations.astype(np.float32)


def normalize_windows(
    windows: np.ndarray, means: np.ndarray, standard_deviations: np.ndarray
) -> np.ndarray:
    """Apply the same training-derived channel statistics to any split."""
    return ((windows - means) / standard_deviations).astype(np.float32)


def build_model() -> tf.keras.Model:
    """Create the intentionally small 1D CNN for 8-class gesture recognition."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(WINDOW_SIZE, len(IMU0_COLUMNS))),
            tf.keras.layers.Conv1D(16, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(32, kernel_size=3, activation="relu"),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_preprocessing(
    means: np.ndarray, standard_deviations: np.ndarray
) -> None:
    """Persist preprocessing constants and the source-label mapping for inference."""
    label_mapping = {
        str(source_label): {
            "class_index": LABEL_TO_INDEX[source_label],
            "name": LABEL_NAMES[source_label],
        }
        for source_label in SOURCE_LABELS
    }
    np.savez_compressed(
        PREPROCESSING_PATH,
        channel_means=means,
        channel_standard_deviations=standard_deviations,
        sampling_rate_hz=np.asarray(TARGET_SAMPLING_RATE_HZ),
        window_size=np.asarray(WINDOW_SIZE),
        hop_size=np.asarray(HOP_SIZE),
        label_mapping=np.asarray(json.dumps(label_mapping)),
        channel_names=np.asarray(IMU0_COLUMNS),
    )


def make_representative_dataset(training_windows: np.ndarray):
    """Yield normalized training examples in the converter's expected batch shape."""
    sample_count = min(REPRESENTATIVE_SAMPLE_COUNT, training_windows.shape[0])
    rng = np.random.default_rng(RANDOM_SEED)
    sample_indices = rng.choice(training_windows.shape[0], sample_count, replace=False)

    def representative_dataset():
        for sample_index in sample_indices:
            yield [training_windows[sample_index : sample_index + 1].astype(np.float32)]

    return representative_dataset


def evaluate_tflite(
    interpreter: tf.lite.Interpreter,
    test_windows: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[float, np.ndarray, dict[str, float | int | tuple[int, ...] | type]]:
    """Run INT8 inference with explicit input quantization and output dequantization."""
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_detail["quantization"]
    output_scale, output_zero_point = output_detail["quantization"]
    if input_scale <= 0 or output_scale <= 0:
        raise ValueError("TFLite model is missing valid input/output quantization parameters.")

    predictions = np.empty(test_labels.shape, dtype=np.int64)
    for index, window in enumerate(test_windows):
        quantized_input = np.rint(window / input_scale + input_zero_point)
        quantized_input = np.clip(
            quantized_input, np.iinfo(np.int8).min, np.iinfo(np.int8).max
        ).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], quantized_input[np.newaxis, ...])
        interpreter.invoke()
        quantized_output = interpreter.get_tensor(output_detail["index"])
        dequantized_output = (
            quantized_output.astype(np.float32) - output_zero_point
        ) * output_scale
        predictions[index] = int(np.argmax(dequantized_output, axis=-1)[0])

    accuracy = float(np.mean(predictions == test_labels))
    quantization = {
        "input_shape": tuple(int(value) for value in input_detail["shape"]),
        "input_dtype": input_detail["dtype"],
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zero_point),
        "output_shape": tuple(int(value) for value in output_detail["shape"]),
        "output_dtype": output_detail["dtype"],
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zero_point),
    }
    return accuracy, predictions, quantization


def print_class_distribution(
    split_name: str,
    recording_indices: np.ndarray,
    recordings: list[Recording],
    window_labels: np.ndarray,
) -> None:
    """Print recording and window counts for every gesture class in one split."""
    source_labels = [recordings[int(index)].source_label for index in recording_indices]
    recording_counts = {
        LABEL_NAMES[label]: int(source_labels.count(label)) for label in SOURCE_LABELS
    }
    window_counts = {
        LABEL_NAMES[source_label]: int(np.count_nonzero(window_labels == class_index))
        for class_index, source_label in enumerate(SOURCE_LABELS)
    }
    class_counts = ", ".join(
        f"{name}={recording_counts[name]}/{window_counts[name]}"
        for name in CLASS_NAMES
    )
    print(f"{split_name}: recordings/windows by class: {class_counts}")


def main() -> None:
    """Run the complete recording-level training and INT8 evaluation experiment."""
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    recordings, skipped_recordings = load_valid_recordings()
    if skipped_recordings:
        print(f"Skipped invalid/unpaired recordings: {len(skipped_recordings)}")
        for filename, reason in skipped_recordings[:10]:
            print(f"  {filename}: {reason}")
        if len(skipped_recordings) > 10:
            print("  Additional skipped recordings omitted from this list.")

    splits = split_recordings(recordings)
    datasets = {}
    short_recordings = {}
    for split_name, recording_indices in splits.items():
        windows, labels, short = make_window_dataset(recordings, recording_indices)
        datasets[split_name] = {"windows": windows, "labels": labels}
        short_recordings[split_name] = short
        if short:
            print(
                f"{split_name}: {len(short)} recording(s) shorter than 2 seconds "
                "produced no windows."
            )
        if windows.shape[0] == 0:
            raise ValueError(f"The {split_name} split contains no complete windows.")

    training_windows = datasets["train"]["windows"]
    validation_windows = datasets["validation"]["windows"]
    test_windows = datasets["test"]["windows"]
    training_labels = datasets["train"]["labels"]
    validation_labels = datasets["validation"]["labels"]
    test_labels = datasets["test"]["labels"]

    means, standard_deviations = fit_channel_normalization(training_windows)
    normalized_training = normalize_windows(training_windows, means, standard_deviations)
    normalized_validation = normalize_windows(
        validation_windows, means, standard_deviations
    )
    normalized_test = normalize_windows(test_windows, means, standard_deviations)
    save_preprocessing(means, standard_deviations)

    model = build_model()
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
    )
    model.fit(
        normalized_training,
        training_labels,
        validation_data=(normalized_validation, validation_labels),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=2,
    )
    model.save(FLOAT_MODEL_PATH)

    _, training_accuracy = model.evaluate(
        normalized_training, training_labels, verbose=0
    )
    _, validation_accuracy = model.evaluate(
        normalized_validation, validation_labels, verbose=0
    )
    test_loss, float_test_accuracy = model.evaluate(
        normalized_test, test_labels, verbose=0
    )
    float_predictions = np.argmax(model.predict(normalized_test, verbose=0), axis=1)

    print("\nFloat model test confusion matrix (true rows, predicted columns):")
    print(confusion_matrix(test_labels, float_predictions, labels=np.arange(len(CLASS_NAMES))))
    print("\nFloat model per-class test performance:")
    print(
        classification_report(
            test_labels,
            float_predictions,
            labels=np.arange(len(CLASS_NAMES)),
            target_names=CLASS_NAMES,
            zero_division=0,
            digits=4,
        )
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset(normalized_training)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    INT8_MODEL_PATH.write_bytes(converter.convert())

    interpreter = tf.lite.Interpreter(model_path=str(INT8_MODEL_PATH))
    interpreter.allocate_tensors()
    int8_test_accuracy, int8_predictions, quantization = evaluate_tflite(
        interpreter, normalized_test, test_labels
    )
    print("\nINT8 model test confusion matrix (true rows, predicted columns):")
    print(confusion_matrix(test_labels, int8_predictions, labels=np.arange(len(CLASS_NAMES))))

    for split_name, recording_indices in splits.items():
        print_class_distribution(
            split_name,
            recording_indices,
            recordings,
            datasets[split_name]["labels"],
        )

    float_size_kb = FLOAT_MODEL_PATH.stat().st_size / 1024
    int8_size_kb = INT8_MODEL_PATH.stat().st_size / 1024
    parameter_count = int(model.count_params())

    print("\n=== Exp7 Task 2: Model Training & Compression ===")
    print(f"Valid paired recordings: {len(recordings)}")
    if skipped_recordings:
        print(f"Skipped recordings: {len(skipped_recordings)}")
    print("\nRecordings:")
    print(f"Train: {len(splits['train'])}")
    print(f"Validation: {len(splits['validation'])}")
    print(f"Test: {len(splits['test'])}")
    print("\nWindows:")
    print(f"Train: {training_labels.size}")
    print(f"Validation: {validation_labels.size}")
    print(f"Test: {test_labels.size}")
    print(f"\nInput shape: ({WINDOW_SIZE}, {len(IMU0_COLUMNS)})")
    print(f"Classes: {len(CLASS_NAMES)}")
    print("Labels: " + ", ".join(f"{label}={LABEL_NAMES[label]}" for label in SOURCE_LABELS))
    print("\nFloat Model:")
    print(f"Training Accuracy: {training_accuracy:.2%}")
    print(f"Validation Accuracy: {validation_accuracy:.2%}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Parameters: {parameter_count}")
    print(f"Size: {float_size_kb:.2f} KB")
    print(f"Test Accuracy: {float_test_accuracy:.2%}")
    print("\nINT8 Model:")
    print(f"Size: {int8_size_kb:.2f} KB")
    print(f"Input shape: {quantization['input_shape']}")
    print(f"Input: {quantization['input_dtype'].__name__}")
    print(f"Output shape: {quantization['output_shape']}")
    print(f"Output: {quantization['output_dtype'].__name__}")
    print(f"Test Accuracy: {int8_test_accuracy:.2%}")
    print("\nInput quantization:")
    print(f"Scale: {quantization['input_scale']:.8g}")
    print(f"Zero point: {quantization['input_zero_point']}")
    print("\nOutput quantization:")
    print(f"Scale: {quantization['output_scale']:.8g}")
    print(f"Zero point: {quantization['output_zero_point']}")
    print("\nModel comparison:")
    print("Model       Parameters   File Size KB   Test Accuracy")
    print(
        f"Float       {parameter_count:<12} {float_size_kb:<13.2f} "
        f"{float_test_accuracy:.2%}"
    )
    print(
        f"INT8 TFLite {parameter_count:<12} {int8_size_kb:<13.2f} "
        f"{int8_test_accuracy:.2%}"
    )
    print(f"\nPreprocessing saved: {PREPROCESSING_PATH.name}")
    print(f"Float model saved: {FLOAT_MODEL_PATH.name}")
    print(f"INT8 model saved: {INT8_MODEL_PATH.name}")


if __name__ == "__main__":
    main()
