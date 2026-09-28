"""Task 1: simulate a 30 Hz IMU stream and generate sliding windows."""

from pathlib import Path

import numpy as np
import pandas as pd


TARGET_SAMPLING_RATE_HZ = 30
# At 30 Hz, 60 samples cover the intended 2-second model input window.
WINDOW_SIZE = 60
# A 200 ms hop at 30 Hz is 6 samples, so successive windows overlap by 54.
HOP_SIZE = 6

# Change this path to test another single recording. The data/ folder contains
# the original recordings; data_clean/ is not used by this initial pipeline.
RECORDING_PATH = (
    Path(__file__).resolve().parent
    / "rosbag"
    / "data"
    / "data"
    / "rosbag2_2023_02_10-06_45_38_data.csv"
)

TIMESTAMP_COLUMN = "timestamp"
IMU0_COLUMNS = (
    "Imu0_linear_accleration_x",
    "Imu0_linear_accleration_y",
    "Imu0_linear_accleration_z",
    "Imu0_angular_velocity_x",
    "Imu0_angular_velocity_y",
    "Imu0_angular_velocity_z",
)


def load_recording(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load timestamps and the six required IMU0 channels from one CSV."""
    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path)
    required_columns = (TIMESTAMP_COLUMN, *IMU0_COLUMNS)
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required CSV columns: {', '.join(missing_columns)}")
    if frame.empty:
        raise ValueError(f"Recording contains no samples: {csv_path}")

    selected = frame.loc[:, required_columns].apply(pd.to_numeric, errors="coerce")
    values = selected.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad_columns = selected.columns[selected.isna().any()].tolist()
        if not bad_columns:
            bad_columns = [
                column
                for index, column in enumerate(selected.columns)
                if not np.isfinite(values[:, index]).all()
            ]
        raise ValueError(
            "Recording contains NaN or non-finite values in: "
            + ", ".join(bad_columns)
        )

    timestamps = values[:, 0]
    if timestamps.size < 2:
        raise ValueError("At least two timestamped samples are required.")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Timestamps must be strictly increasing.")

    return timestamps, values[:, 1:]


def resample_imu(
    timestamps: np.ndarray,
    imu_values: np.ndarray,
    target_rate_hz: int = TARGET_SAMPLING_RATE_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate IMU channels at uniform target-rate times using source timestamps."""
    if target_rate_hz <= 0:
        raise ValueError("Target sampling rate must be positive.")
    if imu_values.ndim != 2 or imu_values.shape != (timestamps.size, len(IMU0_COLUMNS)):
        raise ValueError("IMU data must have one six-channel row per timestamp.")

    relative_times = timestamps - timestamps[0]
    duration = relative_times[-1]
    # Sampling rates differ across recordings, so use recorded timestamps to
    # interpolate each channel onto a uniform 30 Hz time grid.
    sample_count = int(np.floor(duration * target_rate_hz)) + 1
    resampled_times = np.arange(sample_count, dtype=float) / target_rate_hz
    resampled_values = np.column_stack(
        [
            np.interp(resampled_times, relative_times, imu_values[:, channel])
            for channel in range(imu_values.shape[1])
        ]
    )
    return resampled_times, resampled_values


class RingBuffer:
    """Fixed-capacity sample buffer that returns chronological windows."""

    def __init__(self, capacity: int, hop_size: int) -> None:
        if capacity <= 0 or hop_size <= 0 or hop_size > capacity:
            raise ValueError("Capacity and hop size must be positive; hop cannot exceed capacity.")
        self.capacity = capacity
        self.hop_size = hop_size
        self._storage: np.ndarray | None = None
        self._write_index = 0
        self._sample_count = 0

    def append(self, sample: np.ndarray) -> np.ndarray | None:
        """Add one sample; return a copied window when a hop boundary is reached."""
        sample = np.asarray(sample, dtype=float)
        if sample.ndim != 1:
            raise ValueError("Each buffered sample must be a one-dimensional channel row.")
        if self._storage is None:
            self._storage = np.empty((self.capacity, sample.size), dtype=float)
        elif sample.size != self._storage.shape[1]:
            raise ValueError("All samples in a ring buffer must have the same channel count.")

        self._storage[self._write_index] = sample
        self._write_index = (self._write_index + 1) % self.capacity
        self._sample_count += 1

        if self._sample_count < self.capacity:
            return None
        if (self._sample_count - self.capacity) % self.hop_size != 0:
            return None

        # The ring buffer overwrites its oldest row as each new sample arrives;
        # this ordering restores the current window from oldest to newest.
        start = self._write_index
        indices = (start + np.arange(self.capacity)) % self.capacity
        return self._storage[indices].copy()


def generate_windows(
    resampled_values: np.ndarray,
    window_size: int = WINDOW_SIZE,
    hop_size: int = HOP_SIZE,
) -> np.ndarray:
    """Feed samples one at a time and return all complete overlapping windows."""
    buffer = RingBuffer(capacity=window_size, hop_size=hop_size)
    windows = []
    for sample in resampled_values:
        window = buffer.append(sample)
        if window is not None:
            windows.append(window)

    if not windows:
        return np.empty((0, window_size, resampled_values.shape[1]), dtype=float)
    return np.stack(windows)


def main() -> None:
    """Run the pipeline on the one configured representative recording."""
    timestamps, imu_values = load_recording(RECORDING_PATH)
    duration = float(timestamps[-1] - timestamps[0])
    original_rate_hz = (timestamps.size - 1) / duration
    resampled_times, resampled_values = resample_imu(
        timestamps, imu_values, TARGET_SAMPLING_RATE_HZ
    )
    windows = generate_windows(resampled_values, WINDOW_SIZE, HOP_SIZE)

    print("=== Exp7 Task 1: Streaming IMU Pipeline ===")
    print(f"Recording: {RECORDING_PATH.name}")
    print(f"Original samples: {timestamps.size}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Original sampling rate: {original_rate_hz:.2f} Hz")
    print(f"Resampled sampling rate: {TARGET_SAMPLING_RATE_HZ} Hz")
    print(f"Resampled samples: {resampled_values.shape[0]}")
    print(f"Window size: {WINDOW_SIZE} samples / {WINDOW_SIZE / TARGET_SAMPLING_RATE_HZ:.1f} seconds")
    print(f"Hop size: {HOP_SIZE} samples / {HOP_SIZE / TARGET_SAMPLING_RATE_HZ * 1000:.0f} ms")
    print(f"Generated windows: {windows.shape[0]}")
    if windows.shape[0] == 0:
        print(f"Window shape: (0, {WINDOW_SIZE}, {len(IMU0_COLUMNS)})")
        print("No complete window generated; recording is shorter than 2.0 seconds.")
    else:
        print(f"Window shape: {windows.shape[1:]}")
        print(f"Window batch shape: {windows.shape}")
        print("First window, first 5 samples (6 IMU0 channels):")
        print(windows[0, :5])


if __name__ == "__main__":
    main()
