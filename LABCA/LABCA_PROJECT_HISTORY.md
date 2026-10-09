# LABCA Gesture Recognition System - Technical History

## Overview
This document details the technical evolution of the LABCA/gesture-cmd project, a complete end-to-end gesture recognition wearable system using ESP32/ESP8266 and MPU6050 IMU.

## Phase 0: Initial Exploration and Sensor Verification

### Labca_accelerometer_test
**Purpose:** Raw MPU6050 sensor verification and communication testing
**Key Files:**
- `labca_accelerometer_test/labca_accelerometer_test.ino`
**Activities:**
- Basic I2C communication test with MPU6050
- Raw sensor data output to Serial
- Pin configuration: SDA=21, SCL=22 (ESP32 default)
- Verified sensor responsiveness and data ranges
**Outcome:** Confirmed MPU6050 communication viability at 0x68 I2C address

### First_dry_run
**Purpose:** Initial MPU6050 + OLED display integration test
**Key Files:**
- `first_dry_run/first_dry_run.ino`
**Activities:**
- MPU6050 initialization and data reading
- OLED display (SSD1306) integration for real-time data visualization
- Display shows accelerometer and gyroscope values
- Demonstrated basic sensor-to-display pipeline
**Outcome:** Established foundation for sensor data processing and user feedback

### Esp_test and Esp_test_oled_screen
**Purpose:** ESP32 basic functionality and display testing
**Key Files:**
- Various test sketches in these directories
**Activities:**
- ESP32 GPIO and peripheral testing
- OLED display integration verification
- WiFi and basic communication tests
**Outcome:** Validated ESP32 platform readiness for sensor integration

## Phase 1: Firmware Development (Gesture-cmd Firmware)

### IMU Reader Development
**Key File:** `LABCA/gesture-cmd/firmware/src/imu_reader.h`
**Purpose:** Reusable MPU6050 I2C driver with register-level access
**Key Features:**
- Direct register access (no external libraries)
- Register definitions: PWR_MGMT_1=0x6B, ACCEL_CONFIG=0x1C, GYRO_CONFIG=0x1B, ACCEL_XOUT_H=0x3B
- Wake-up sequence: Write 0x00 to PWR_MGMT_1
- Configuration: ±2g accel (ACCEL_CONFIG=0x00), ±250°/s gyro (GYRO_CONFIG=0x00)
- Scaling factors: ACCEL_SCALE=16384.0, GYRO_SCALE=131.0f
- Functions: initIMU(), readIMU(&ax,&ay,&az,&gx,&gy,&gz)
**Design Choice:** Direct register access chosen over libraries (e.g., Adafruit MPU6050) to avoid compilation issues and minimize dependencies

### Data Collection Firmware (Phase 01)
**Key File:** `LABCA/gesture-cmd/firmware/src/data_collection.cpp`
**Purpose:** Serial logging of IMU data for dataset creation
**Key Features:**
- I2C communication via Wire.h
- MPU6050 initialization and configuration (±2g accel, ±250°/s gyro)
- 50Hz sampling rate (20ms interval)
- 50-sample windowing (1 second duration)
- Serial command interface:
  - 's': Start recording window
  - 'x': Discard current window
- Output format: CSV lines `timestamp_ms,ax,ay,az,gx,gy,gz`
- Window markers: `# WINDOW_START`, `# WINDOW_END`, `# WINDOW_DISCARD`
- Window buffering: Automatic stop after 50 samples
**Design Decisions:**
- UART serial chosen for simplicity and debugging visibility
- Fixed window size enables consistent ML input dimensions
- Manual start/stop allows gesture timing variability within window
- No on-device processing - pure data collection

### UDP IMU Streaming Firmware (Phase 03)
**Key File:** `LABCA/gesture-cmd/firmware/src/main.cpp`
**Purpose:** Wireless transmission of IMU windows to PC for real-time inference
**Key Features:**
- WiFi connection (hardcoded credentials)
- UDP packet transmission to PC
- Packet format (1208 bytes):
  - 4 bytes: uint32 sequence number
  - 4 bytes: uint32 timestamp_ms
  - 1200 bytes: float32[50][6] window (ax,ay,az,gx,gy,gz)
- Uses imu_reader.h and window_buffer.h
- 50Hz sampling with sliding window
- Continuous transmission - no ML on device
**Design Choices:**
- UDP selected for low latency and simplicity (no acknowledgment overhead)
- Binary format chosen for efficiency over CSV
- Sequence numbers enable loss detection
- Offloads inference to PC for easier debugging and iteration
- Window_buffer.h provides circular buffer for efficient sliding window

## Phase 2: Server-side ML Pipeline

### Data Collection and Labeling
**Key File:** `LABCA/gesture-cmd/server/src/training/gesture_logger.py`
**Purpose:** Gesture-labeled data collection from serial stream
**Key Features:**
- Serial port configuration and data parsing
- Window-based labeling with 's'/'x' command interface
- CSV output with format: `label,sample_id,timestamp_ms,sample,ax,ay,az,gx,gy,gz`
- Automatic window numbering and boundary marking
- Visual feedback during collection
**Workflow:**
1. User sends 's' via serial to start window
2. System collects 50 samples @ 50Hz
3. User performs gesture during window
4. Window auto-saves after 50 samples or user sends 'x' to discard
5. Data labeled with specified gesture name

### Preprocessing Pipeline
**Key File:** `LABCA/gesture-cmd/server/src/training/preprocess.py`
**Purpose:** Transform raw CSV into model-ready tensors
**Key Steps:**
1. **Load Dataset:** Read CSV with pandas
2. **Validate Windows:** Ensure 50 samples/window, label consistency
3. **Build Windows:** Group by (label, sample_id) → shape (N, 50, 6)
4. **Label Encoding:** String labels → integers via LabelEncoder
5. **Train/Val/Test Split:** Stratified 70/15/15 split
6. **Normalization:** StandardScaler fitted ONLY on training data
   - Per-channel: (x - μ_train) / σ_train
   - Applied identically to all sets
7. **Save Artifacts:**
   - X_train.npy, X_val.npy, X_test.npy
   - y_train.npy, y_val.npy, y_test.npy
   - label_map.json (gesture ↔ index mapping)
   - scaler.pkl (fitted StandardScaler)
8. **Sanity Checks:** Sample waveform plots saved as PNGs

**Critical Anti-Leakage Measures:**
- Window-level grouping prevents sample splitting across sets
- StandardScaler fitted ONLY on training data
- Stratified splits maintain class distribution
- No sample-level modifications - corrupted windows rejected entirely

### Model Training
**Key File:** `LABCA/gesture-cmd/server/src/training/train_model.py`
**Purpose:** Train lightweight 1D-CNN for gesture recognition
**Architecture:**
```
Input: (50, 6)
→ Conv1D(32, k=5) + BatchNorm + MaxPooling1D + Dropout(0.3)
→ Conv1D(64, k=3) + BatchNorm + MaxPooling1D + Dropout(0.3)
→ Conv1D(64, k=3) + BatchNorm + Dropout(0.3)
→ GlobalAveragePooling1D
→ Dense(64, ReLU) + Dropout(0.3)
→ Dense(6, Softmax)
```
**Features:**
- ~24K parameters (~15KB float32 weights)
- Adam optimizer (lr=1e-3)
- Sparse categorical crossentropy loss
- Early stopping (patience=15)
- ReduceLROnPlateau (factor=0.5, patience=8)
- Model checkpointing
- Training history visualization
**Output:** `gesture_model.keras` (full Keras model)

### Quantization and Export
**Key File:** `LABCA/gesture-cmd/server/src/training/quantize_export.py`
**Purpose:** Convert model to deployment-ready formats
**Process:**
1. Load trained Keras model
2. Create TFLite float32 baseline
3. Create INT8 quantized model:
   - Optimizations: [tf.lite.Optimize.DEFAULT]
   - Representative dataset: 200 samples from X_train
   - Supported ops: [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
   - Input/output type: tf.int8
4. Evaluate both models on test set
5. Export artifacts:
   - `gesture_model_float32.tflite` (baseline)
   - `gesture_model_int8.tflite` (deployable)
   - `gesture_model.h` (C header for Arduino)
   - `quantization_report.txt` (accuracy/size comparison)
**Results from quantization_report.txt:**
- Size: 107,044 bytes → 41,448 bytes (61.3% compression)
- Accuracy: 99.39% → 99.39% → 99.39% (0.00% drop)
- Label map: {0:flick_down, 1:flick_up, 2:idle, 3:wave_left, 4:wave_right, 5:(missing?)}
**Note:** Label map shows 5 classes, suggesting fist_hold was removed during development

## Phase 3: Inference and Interaction

### Inference Server
**Key File:** `LABCA/gesture-cmd/server/src/server/inference_server.py`
**Purpose:** Bridge UDP packets → TFLite inference → MQTT
**Features:**
- UDP packet receiver (port 5005)
- TFLite interpreter for INT8 model
- Same preprocessing as training (using scaler.pkl)
- Gesture classification and confidence scoring
- MQTT publisher to "wearable/gesture" topic
- Configurable threshold (default 0.85)
- Sequence number tracking for loss detection

### HCI Controller
**Key File:** `LABCA/gesture-cmd/server/src/hci/hci_controller.py`
**Purpose:** MQTT gestures → keyboard shortcuts
**Features:**
- MQTT subscriber to "wearable/gesture"
- Gesture-to-shortcut mapping (customizable)
- Multiple backend support:
  - uinput (Wayland, requires sudo)
  - X11 (XLib)
  - Console (debugging)
- Default mappings:
  - wave_right: Ctrl+Right (next desktop/tab)
  - wave_left: Ctrl+Left (previous desktop/tab)
  - flick_up: Super (show all windows)
  - flick_down: Ctrl+Alt+D (show desktop)
  - fist_hold: Ctrl+L (focus address/lock) - suggests this class existed
  - wrist_rotate: Alt+Tab (switch application)
  - idle: (none - suppressed)
- Configuration via bindings.json file
- Wayland/X11 compatibility handling

### Dashboard (Optional)
**Key File:** `LABCA/gesture-cmd/server/dashboard/gesture_dashboard.html`
**Purpose:** Real-time gesture monitoring UI
**Features:**
- WebSocket connection to inference server
- Live gesture visualization
- Confidence metrics
- Connection status
- Runs on port 9001

## Hardware Configuration (Final)

Documented in `LABCA/Hardware_confs.md`:
- **Microcontroller:** ESP32-WROOM
- **IMU:** MPU6050
- **Communication:** I²C (SDA=GPIO21, SCL=GPIO22)
- **Sampling Rate:** 50 Hz
- **Window Size:** 50 samples (1 second)
- **Sensor Channels:** ax, ay, az, gx, gy, gz
- **Accelerometer Range:** ±2g
- **Gyroscope Range:** ±1000°/s (chosen to avoid saturation during wrist rotation)
- **Number of Classes:** 6 (idle, wave_left, wave_right, flick_up, flick_down, wrist_rotate)
- **Rationale:** Sensor integrity, class separability, reproducibility

## Project Evolution Summary

1. **Sensor Verification:** Confirm MPU6050 communication and data viability
2. **Firmware Development:** 
   - Phase 01: Serial data collection (UART CSV)
   - IMU reader library (direct register access)
   - Phase 03: UDP wireless streaming to PC
3. **ML Pipeline:**
   - Data labeling via serial commands
   - Windowing (50 samples = 1 sec)
   - Preprocessing (train-only StandardScaler)
   - 1D-CNN model (~24K params)
   - INT8 quantization (61% size reduction, 0% accuracy drop)
4. **Deployment:**
   - UDP→TFLite→MQTT inference server
   - MQTT→keyboard shortcuts HCI controller
   - Optional web dashboard

## Key Technical Decisions and Rationales

| Decision | Alternative | Reason |
|----------|-------------|--------|
| Direct Register Access | Adafruit MPU6050 library | Avoided compilation issues, minimized dependencies |
| UART for Data Collection | Immediate processing | Simplicity, debugging visibility, separates concerns |
| UDP for Transmission | TCP, MQTT direct | Lower overhead, simplicity, application-level reliability sufficient |
| PC-based Inference | On-device (TFLite Micro) | Easier debugging/iteration; on-device planned for future |
| 1D-CNN Architecture | Dense/RNN/Tree models | Temporal feature extraction, parameter efficiency |
| Windowed Recording | Continuous/Event-triggered | Uniform input size, prevents positional memorization |
| StandardScaler (train-only) | Global fitting, MinMax | Prevents data leakage, centers/scale features appropriately |
| INT8 Quantization | Float32, dynamic range | 4x size reduction with minimal accuracy impact |
| Global Average Pooling | Flattening after Conv | Dramatically reduces parameters before Dense layers |
| 6-axis IMU | Accel-only, 9-axis | Gyroscope essential for rotation detection; magnetometer unnecessary indoors |

This LABCA/gesture-cmd system represents a complete, well-documented Edge AI pipeline from sensor data collection to user interaction, with careful attention to reproducibility, data integrity, and deployment considerations.