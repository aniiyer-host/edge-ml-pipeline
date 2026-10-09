# LABCA Gesture Recognition System - Technical History

## System Overview
The LABCA/gesture-cmd project is a complete end-to-end gesture recognition wearable system that recognizes hand gestures using an ESP32/ESP8266 + MPU6050 IMU sensor and translates them into keyboard shortcuts on a PC. The system consists of firmware for data collection/transmission, a server-side machine learning pipeline, and interface components for real-time interaction.

## Phase 0: Sensor Validation and Exploration

### Initial Hardware Verification
The project began with verification of the MPU6050 sensor connectivity and data quality:

- **labca_accelerometer_test/**: Confirmed basic I2C communication with MPU6050 at address 0x68
  - Used SDA=GPIO21, SCL=GPIO22 (ESP32 default pins)
  - Outputted raw accelerometer and gyroscope values to Serial
  - Verified sensor responsiveness without relying on external libraries

- **first_dry_run/**: Initial integration test with OLED display
  - Combined MPU6050 sensor reading with SSD1306 OLED display
  - Showed real-time sensor values on display (AX, AY, AZ, GX, GY, GZ)
  - Demonstrated basic sensor-to-user feedback pipeline
  - Used pins: SDA=21, SCL=22, OLED_ADDR=0x3C

### Key Insight from Early Tests
These initial experiments confirmed:
1. MPU6050 communication viability via I2C
2. Sensor data quality sufficient for gesture recognition
3. ESP32 platform compatibility with both sensor and display
4. Foundation for more sophisticated firmware development

## Phase 1: Firmware Architecture Development

The firmware evolved through three distinct phases, each building upon the previous:

### Phase 01: Data Collection Firmware
**File:** `LABCA/gesture-cmd/firmware/src/data_collection.cpp`
**Purpose:** Serial logging of IMU data for dataset creation

**Key Features:**
- Direct I2C communication using Wire.h library
- MPU6050 initialization sequence:
  1. Wake from sleep: Write 0x00 to PWR_MGMT_1 (0x6B)
  2. Configure accelerometer: ±2g range via ACCEL_CONFIG (0x1C) = 0x00
  3. Configure gyroscope: ±250°/s range via GYRO_CONFIG (0x1B) = 0x00
- 50Hz sampling rate (20ms interval) using millis() timing
- 50-sample fixed windows (1.0 second duration)
- Serial command interface:
  - 's': Start recording window (outputs `# WINDOW_START`)
  - 'x': Discard current window (outputs `# WINDOW_DISCARD`)
- CSV output format: `timestamp_ms,ax,ay,az,gx,gy,gz` (4 decimal places)
- Window completion detection: Automatic stop after 50 samples (`# WINDOW_END`)
- Uses imu_reader.h for sensor access

**Design Decisions:**
- UART serial chosen for simplicity and debugging visibility during data collection
- Fixed window size enables consistent input dimensions for machine learning
- Manual start/stop control allows natural gesture timing variability within window
- Pure data collection approach - no on-device processing to keep firmware simple

### IMU Reader Library
**File:** `LABCA/gesture-cmd/firmware/src/imu_reader.h`
**Purpose:** Reusable, lightweight MPU6050 I2C driver

**Key Features:**
- Register-level access (avoiding external libraries like Adafruit_MPU6050)
- Register definitions:
  - MPU_ADDR = 0x68
  - PWR_MGMT_1 = 0x6B (power management)
  - ACCEL_CONFIG = 0x1C (accelerometer configuration)
  - GYRO_CONFIG = 0x1B (gyroscope configuration)
  - ACCEL_XOUT_H = 0x3B (accelerometer data starting register)
- Configuration functions:
  - initIMU(): Wakes sensor and sets ±2g accel, ±250°/s gyro
  - readIMU(): Reads all 6 axes and converts to physical units
- Scaling factors:
  - ACCEL_SCALE = 16384.0f (±2g range → 2^15 LSB/g)
  - GYRO_SCALE = 131.0f (±250°/s range → 2^15 LSB/(°/s))
- Uses int16_t raw data buffer for efficient I2C reading

**Design Choice - Direct Register Access:**
Chosen over external libraries due to:
1. Previous compile issues with Adafruit_MPU6050 library
2. Desire for minimal dependencies and firmware size
3. Transparency in sensor configuration and data reading
4. Sufficient simplicity for this sensor's straightforward register map

### Phase 03: UDP IMU Streaming Firmware
**File:** `LABCA/gesture-cmd/firmware/src/main.cpp`
**Purpose:** Wireless transmission of IMU data to PC for real-time inference

**Key Features:**
- WiFi connection with hardcoded credentials (WIFI_SSID, WIFI_PASSWORD)
- UDP packet transmission to inference server on PC
- Optimized packet format (1208 bytes total):
  - [0-3]: uint32 sequence number (for loss detection)
  - [4-7]: uint32 timestamp_ms (window start time)
  - [8-1207]: float32[50][6] window data (50 timesteps × 6 channels)
    - Order: [ax,ay,az,gx,gy,gz] for each timestep
    - Total: 50 × 6 × 4 bytes = 1200 bytes
- Uses imu_reader.h for sensor access
- Uses window_buffer.h for efficient sliding window management
- 50Hz sampling rate (SAMPLE_INTERVAL_MS = 20)
- Continuous operation: samples → fills window → transmits → repeats
- No ML processing on device - offloads inference to PC

**Design Choices:**
- UDP selected over TCP for lower overhead and latency
  - No connection establishment/maintenance
  - No acknowledgment delay
  - Application-level reliability sufficient (sequence numbers detect loss)
- Binary format chosen over CSV for transmission efficiency
  - 1208 bytes binary vs ~6KB+ CSV for same data
  - Native float32 format matches TFLite interpreter expectations
- Offloading inference to PC enables:
  - Easier debugging and iteration
  - Access to greater computational resources
  - Simpler firmware (no TensorFlow Lite Micro initially)
  - Path forward for eventual on-device migration

### Supporting Components
**Window Buffer (`window_buffer.h`):**
- Circular buffer implementation for efficient sliding window
- Fixed size: 50 samples × 6 channels
- Overwrite oldest data when new data arrives
- Provides contiguous data blocks for transmission

## Phase 2: Server-Side Machine Learning Pipeline

The server-side components transform raw IMU data into a deployable gesture recognition system.

### Data Collection and Labeling
**File:** `LABCA/gesture-cmd/server/src/training/gesture_logger.py`
**Purpose:** Gesture-labeled data collection from serial stream

**Process:**
1. Opens serial connection to ESP32 firmware
2. Parses incoming CSV-format data lines
3. Implements window-based labeling protocol:
   - User sends 's' to start recording window
   - System collects exactly 50 samples @ 50Hz
   - User performs gesture during this window
   - Window auto-saves after 50 samples OR user sends 'x' to discard
4. Outputs labeled CSV with format:
   `label,sample_id,timestamp_ms,sample,ax,ay,az,gx,gy,gz`
5. Includes window boundary markers in comments for human readability
6. Provides real-time feedback during collection

**Key Features:**
- Automatic window numbering (sample_id increments)
- Label association with each complete window
- Flexible timing: gestures can occur at any position within window
- Discard mechanism for mistimed or corrupted attempts
- Human-readable format with metadata comments

### Preprocessing Pipeline
**File:** `LABCA/gesture-cmd/server/src/training/preprocess.py`
**Purpose:** Transforms raw CSV into model-ready tensors with rigorous anti-leakage measures

**Critical Design Principle:** Prevent data leakage at all costs - never allow information from validation/test sets to influence training

**Processing Steps:**

1. **Data Loading and Validation**
   - Load CSV with pandas
   - Verify exact 50 samples per window
   - Confirm label consistency within each window
   - Validate sequential sample numbering (0-49)
   - Check chronological timestamp ordering
   - Ensure all sensor values are valid numbers
   - Reject entire window if ANY validation fails (no sample-level editing)

2. **Window Extraction**
   - Groups data by (`label`, `sample_id`) 
   - Each group forms tensor of shape (50, 6)
   - Returns:
     - X: numpy array (N_windows, 50, 6)
     - y_strings: list of gesture labels per window

3. **Label Encoding**
   - Uses sklearn LabelEncoder
   - Creates bidirectional mapping: label_map.json
   - Transforms labels to integers (0,1,2,...)

4. **Train/Validation/Test Split**
   - Stratified split maintains class distribution
   - First split: 85% (train+val) / 15% (test)
   - Second split: 70% train / 15% val from the 85% portion
   - Final split: 70% train, 15% val, 15% test
   - Uses random_state=42 for reproducibility
   - Stratification ensures similar gesture proportions in each set

5. **Feature Normalization (CRITICAL STEP)**
   - **Fit StandardScaler ONLY on training data** (X_train)
     - Computes mean and standard deviation PER CHANNEL
     - ax, ay, az, gx, gy, gz each normalized independently
   - **Transform all sets using SAME parameters**
     - Training: (X_train - μ_train) / σ_train
     - Validation: (X_val - μ_train) / σ_train  
     - Test: (X_test - μ_train) / σ_train
   - Formula applied per feature: x_normalized = (x - μ) / σ
   - Reshapes to (N*50, 6) for statistics computation, then back

6. **Artifact Persistence**
   - Saves normalized arrays:
     - X_train.npy, X_val.npy, X_test.npy (shape: N×50×6)
     - y_train.npy, y_val.npy, y_test.npy (shape: N,)
   - Saves metadata:
     - label_map.json (gesture ⇄ index mapping)
     - scaler.pkl (fitted StandardScaler for identical preprocessing)
   - Optional: Sample waveform plots for sanity checking

**Why This Prevents Leakage:**
- Window-level grouping: Never splits samples from same window across sets
- Scalar fitting: Uses ONLY training data to compute μ and σ
- Stratified splits: Maintains class representation without peeking at labels
- No peeking: Validation/test data never influences training parameters

### Model Training
**File:** `LABCA/gesture-cmd/server/src/training/train_model.py`
**Purpose:** Trains lightweight 1D-CNN optimized for Edge AI deployment

**Architecture Design Goals:**
- Target: <25K parameters (~15KB float32 weights)
- Temporal feature extraction for 6-channel IMU data
- Compatibility with quantization and microcontroller deployment
- Robustness to limited dataset size

**Architecture:**
```
Input: (50, 6) [50 timesteps, 6 IMU channels: ax,ay,az,gx,gy,gz]
│
├─ Block 1: Feature Extraction
│   ├─ Conv1D(32 filters, kernel_size=5, padding='same', activation='relu')
│   ├─ BatchNormalization()
│   ├─ MaxPooling1D(pool_size=2)
│   └─ Dropout(rate=0.3)
│   → Output: (25, 32)
│
├─ Block 2: Feature Refinement  
│   ├─ Conv1D(64 filters, kernel_size=3, padding='same', activation='relu')
│   ├─ BatchNormalization()
│   ├─ MaxPooling1D(pool_size=2)
│   └─ Dropout(rate=0.3)
│   → Output: (12, 64)
│
├─ Block 3: High-Level Features
│   ├─ Conv1D(64 filters, kernel_size=3, padding='same', activation='relu')
│   ├─ BatchNormalization()
│   └─ Dropout(rate=0.3)
│   → Output: (12, 64)
│
├─ Temporal Pooling
│   └─ GlobalAveragePooling1D()
│   → Output: (64,)  [One value per feature map]
│
└─ Classification Head
    ├─ Dense(64, activation='relu')
    ├─ Dropout(rate=0.3)
    └─ Dense(n_classes, activation='softmax')
    → Output: (n_classes,) [Probability distribution]
```

**Parameter Count Estimate:**
- Block 1 Conv1D: (5×6×32) + 32 = 992
- Block 1 BatchNorm: 2×32 = 64
- Block 2 Conv1D: (3×32×64) + 64 = 6,208
- Block 2 BatchNorm: 2×64 = 128
- Block 3 Conv1D: (3×64×64) + 64 = 12,352
- Block 3 BatchNorm: 2×64 = 128
- Dense(64): (64×64) + 64 = 4,160
- Dense(6): (64×6) + 6 = 390
- **Total:** ~24,422 parameters (~15KB float32)

**Training Configuration:**
- Optimizer: Adam (learning rate = 1e-3)
- Loss: Sparse categorical crossentropy (efficient for integer labels)
- Metrics: Accuracy
- Batch size: 32
- Epochs: Up to 80
- Callbacks:
  - EarlyStopping: Monitor val_accuracy, patience=15, restore_best_weights
  - ReduceLROnPlateau: Monitor val_loss, factor=0.5, patience=8, min_lr=1e-5
  - ModelCheckpoint: Save best model by val_accuracy

**Outputs:**
- `gesture_model.keras`: Full Keras model (float32)
- `training_history.png`: Loss/accuracy curves
- Evaluation on test set: accuracy, confusion matrix, classification report

### Quantization and Export
**File:** `LABCA/gesture-cmd/server/src/training/quantize_export.py`
**Purpose:** Converts trained model to deployment-ready formats with size reduction

**Quantization Strategy: Full Integer INT8 Post-Training Quantization**
- Why INT8? 4x size reduction vs float32 with minimal accuracy impact
- Why full integer? Maximizes compression and MCU compatibility
- Why post-training? Simpler than quantization-aware training for initial deployment

**Process:**
1. Load trained Keras model (`gesture_model.keras`)
2. Create float32 TFLite baseline (no quantization)
3. Create INT8 quantized model:
   - `converter.optimizations = [tf.lite.Optimize.DEFAULT]`
   - `converter.representative_dataset = lambda: representative_dataset(X_train)`
     - Uses 200 random training samples to calibrate activation ranges
   - `converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]`
   - `converter.inference_input_type = tf.int8`
   - `converter.inference_output_type = tf.int8`
4. Evaluate both models on held-out test set
5. Export deployment artifacts:

**Generated Files:**
- `gesture_model_float32.tflite`: Baseline for comparison (107,044 bytes)
- `gesture_model_int8.tflite`: Deployable quantized model (41,448 bytes)
- `gesture_model.h`: C header array for Arduino/ESP32 embedding
- `quantization_report.txt`: Detailed size/accuracy comparison

**Quantization Results (from report):**
- Size reduction: 107,044 → 41,448 bytes (61.3% smaller)
- Accuracy: 
  - Keras float32: 99.39%
  - TFLite float32: 99.39% 
  - TFLite INT8: 99.39%
  - Accuracy drop: 0.00%
- Label map confirmed: {0:flick_down, 1:flick_up, 2:idle, 3:wave_left, 4:wave_right}
  - Note: 5 classes shown suggests fist_hold was removed during development

**Embedded Usage:**
- `gesture_model.h` contains:
  - Model length in bytes
  - PROGMEM array of hex bytes representing TFLite flatbuffer
  - Ready to `#include` in Arduino firmware
- Enables true on-device inference with TensorFlow Lite Micro

## Phase 3: Real-Time Inference and Interaction

### Inference Server
**File:** `LABCA/gesture-cmd/server/src/server/inference_server.py`
**Purpose:** Bridges UDP packets → TFLite inference → MQTT

**Workflow:**
1. Listens for UDP packets on port 5005
2. Parses packet format:
   - First 4 bytes: uint32 sequence number
   - Next 4 bytes: uint32 timestamp_ms
   - Remaining 1200 bytes: float32[50][6] window data
3. Applies IDENTICAL preprocessing as training:
   - Loads `scaler.pkl` (fitted StandardScaler)
   - Normalizes: (window - μ_train) / σ_train per channel
   - Ensures feature order: [ax,ay,az,gx,gy,gz]
4. Runs TFLite INT8 inference:
   - Quantizes input float32 → int8 using layer-specific scale/zero_point
   - Executes integer arithmetic
   - Dequantizes output int8 → float32 using layer-specific scale/zero_point
5. Applies softmax to get class probabilities
6. Publishes result to MQTT topic "wearable/gesture":
   - Payload: JSON with gesture label and confidence
   - Format: `{"gesture": "wave_right", "confidence": 0.96}`
7. Tracks sequence numbers for loss detection/reporting

**Key Features:**
- Configurable confidence threshold (default 0.85)
- Same preprocessing pipeline guarantees train/test consistency
- UDP sequence numbers detect transmission loss
- Low-latency design for responsive interaction
- Python typing and modular structure for maintainability

### HCI Controller (Human-Computer Interface)
**File:** `LABCA/gesture-cmd/server/src/hci/hci_controller.py`
**Purpose:** Translates MQTT gestures → keyboard shortcuts on host PC

**Workflow:**
1. Subscribes to MQTT topic "wearable/gesture"
2. Receives JSON payloads: `{"gesture": "...", "confidence": ...}`
3. Checks confidence against threshold (default 0.85)
4. Maps recognized gesture to keyboard shortcut
5. Executes shortcut on host operating system
6. Provides feedback/logging for debugging

**Backends Supported:**
- **uinput**: Linux kernel module for injecting keyboard events
  - Required for Wayland (needs sudo)
  - Most reliable method
- **X11**: Uses XLib to send keyboard events
  - Works on Xorg sessions
  - No special permissions needed
- **console**: Prints to terminal (debugging only)

**Default Gesture Mappings:**
- `wave_right`: Ctrl+Right (next desktop/tab)
- `wave_left`: Ctrl+Left (previous desktop/tab) 
- `flick_up`: Super (show all windows - e.g., GNOME Activities overview)
- `flick_down`: Ctrl+Alt+D (show desktop)
- `fist_hold`: Ctrl+L (focus address bar/lock screen) 
  - Note: Suggests this class existed during development but may have been removed
- `wrist_rotate`: Alt+Tab (switch application)
- `idle`: (none - suppressed to prevent false triggers)

**Customization:**
- Users can create `bindings.json` file to override defaults
- Example:
  ```json
  {
    "wave_right": {"keys": ["ctrl", "pagedown"], "description": "Next browser tab"},
    "flick_up": {"keys": ["f11"], "description": "Toggle fullscreen"}
  }
  ```

### Optional Dashboard
**File:** `LABCA/gesture-cmd/server/dashboard/gesture_dashboard.html`
**Purpose:** Real-time gesture monitoring and debugging

**Features:**
- Connects to inference server via WebSocket (port 9001)
- Displays live gesture predictions and confidence
- Shows connection status and packet loss statistics
- Provides visual feedback during development/testing
- Runs in standard web browser (no installation needed)

## Complete System Architecture

### Data Flow
```
[ESP32 + MPU6050]
        │ Samples IMU at 50Hz (20ms)
        △ 50 samples = 1 second window
        │
[UDP Packet] ───────────────▶ [PC: Inference Server]
        │ 1208 bytes: [seq][timestamp][float32[50][6]]
        △ Receives packets, applies SAME standardization as training
        │
[TFLite INT8 Inference] ─────▶ [Softmax → Gesture + Confidence]
        │
[MQTT Publish] ─────────────▶ [PC: HCI Controller]
        │ JSON: {"gesture":"wave_right","confidence":0.96}
        △ Confidence > threshold? → Execute keyboard shortcut
        │
[OS Keyboard Events] ───────▶ [Application Response]
        │ e.g., Ctrl+Right → Switch to next desktop
        ▼
[User Perceives Action]
```

### Critical Anti-Leakage Measures Throughout System
1. **Window Integrity**: Never splits samples from same window across dataset splits
2. **Scalar Fitting**: StandardScaler fitted ONLY on training data
3. **Preprocessing Consistency**: Identical normalization applied at inference
4. **Stratified Splits**: Maintains class distribution during train/val/test split
5. **No Sample-Level Editing**: Corrupted windows rejected entirely
6. **Quantization Calibration**: Uses training data only for INT8 range determination

### Hardware-Software Contract
**Firmware → Server Interface:**
- UDP packet format: 1208 bytes
  - Bytes 0-3: uint32 sequence number
  - Bytes 4-7: uint32 timestamp_ms  
  - Bytes 8-1207: float32[50][6] in order [ax,ay,az,gx,gy,gz] per timestep
- Sampling rate: 50 Hz (implied by window size and timing)
- Window size: Exactly 50 samples
- Data order: Fixed channel sequence (ax,ay,az,gx,gy,gz)

**Server → Firmware (Future On-Device Deployment):**
- Will require identical:
  - Window size (50 samples)
  - Sampling rate (50 Hz)
  - Sensor configuration (±2g accel, ±1000°/s gyro - see Hardware_confs.md)
  - Preprocessing parameters (μ_train, σ_train from scaler.pkl)
  - Model input format (float32[50][6] → quantized using model's scale/zero_point)
  - Output interpretation (softmax probabilities → argmax class)

## Key Technical Decisions and Rationales

### Sensor Configuration
| Decision | Reason |
|----------|--------|
| **MPU6050 ±2g Accelerometer** | Sufficient range for human gestures (~0.0001g/LSB resolution), avoids noise from higher ranges |
| **MPU6050 ±1000°/s Gyroscope** | Empirically chosen to prevent saturation during wrist rotation (tested up to ~±822°/s without clipping) |
| **50Hz Sampling** | Nyquist-appropriate for human motion (<25Hz), balances power/data needs |
| **50-Sample Windows** | 1-second duration captures full gesture dynamics while maintaining low latency |
| **I2C Communication** | Minimal pins (2 signals), adequate bandwidth (300 values/sec), simple master-slave protocol |
| **ESP32-WROOM** | Dual-core headroom for potential on-device ML, integrated WiFi, sufficient RAM/flash |

### Machine Learning Choices
| Decision | Reason |
|----------|--------|
| **1D-CNN Architecture** | Extracts temporal features efficiently; dense/RNN inadequate for parameter constraints |
| **Progressive Filter Sizes** | k=5 first layer captures longer patterns (~100ms), k=3 later layers refine details |
| **Filter Progression 32→64→64** | Starts broad, increases representational power, maintains before pooling |
| **BatchNorm + Dropout(0.3)** | Complementary regularization: stabilizes training + prevents overfitting |
| **GlobalAveragePooling1D** | Dramatically reduces parameters before Dense layers (key to <25K target) |
| **Dense(64) → Dense(n_classes)** | Sufficient capacity for 6-class classification with intermediate representation |
| **Adam Optimizer (lr=1e-3)** | Reliable convergence with minimal tuning |
| **Sparse Categorical Crossentropy** | Memory-efficient for integer labels |
| **Stratified 70/15/15 Split** | Adequate data for each split while maintaining class distribution |

### Deployment Pipeline
| Decision | Reason |
|----------|--------|
| **INT8 Quantization** | 4x size reduction (~15KB → ~4KB) with 0% measured accuracy drop |
| **Representative Dataset Calibration** | 200 samples determines optimal activation ranges for quantization |
| **C Header Export** | Enables direct embedding in Arduino/ESP32 firmware without file system |
| **PC-First Inference** | Easier debugging/iteration; validates pipeline before on-device port |
| **UDP Binary Transmission** | 1208 bytes vs 6KB+ CSV; native float32 matches TFLite expectations |
| **MQTT for PC Communication** | Decouples inference from HCI; enables multiple subscribers (dashboard, logging) |

### Data Collection Protocol
| Decision | Reason |
|----------|--------|
| **Window-Based Labeling** | Enforces uniform input size; allows gesture timing variability within window |
| **Serial 's'/'x' Commands** | Simple, reliable start/discard mechanism |
| **Automatic Window Closure** | Prevents buffer overflow; ensures consistent sample count |
| **Entire Window Rejection** | Preserves data integrity principle: never modify raw training data |
| **Variable Gesture Timing** | Prevents model from learning positional memorization; encourages temporal invariance |
| **Natural Variation Encouraged** | Different speeds, amplitudes, starting/ending positions improve robustness |

## Evolution Summary

The LABCA/gesture-cmd system evolved through clearly defined phases:

1. **Sensor Validation** (labca_accelerometer_test/, first_dry_run/): 
   - Confirmed MPU6050 communication and data quality
   - Verified ESP32 compatibility with sensor and display

2. **Firmware Development** (LABCA/gesture-cmd/firmware/):
   - Phase 01: UART serial data collection (debugging-focused)
   - IMU reader library: Direct register access for transparency and minimal dependencies
   - Phase 03: UDP wireless streaming to PC (enables real-time inference)

3. **Server-Side ML Pipeline** (LABCA/gesture-cmd/server/):
   - Data labeling: Gesture_logger.py for supervised collection
   - Preprocessing: Rigorous anti-leakage windowing and standardization
   - Model training: Lightweight 1D-CNN (~24K params) optimized for Edge AI
   - Quantization: INT8 conversion with 4x size reduction and no accuracy loss
   - Artifact export: TFLite model + C header for embedded deployment

4. **Real-Time Interaction** (LABCA/gesture-cmd/server/src/):
   - Inference server: UDP → TFLite → MQTT bridge
   - HCI controller: MQTT → keyboard shortcuts execution
   - Optional dashboard: Real-time monitoring and debugging

## Key Strengths of This Approach

1. **Reproducibility**: Rigorous data pipeline with zero tolerance for leakage
2. **Modularity**: Clear separation of concerns (firmware, server, interface)
3. **Deployment Awareness**: Architecture designed with quantization and MCU constraints from start
4. **Iterative Development**: PC-based inference enables rapid prototyping before on-device port
5. **User-Centered Design**: Natural gesture timing variation encouraged during collection
6. **Transparency**: Direct register access, documented decisions, explainable architecture

## Path Forward to True Edge Deployment

The current PC-based inference architecture establishes a complete pipeline that can eventually migrate to true on-device operation:

1. **Short Term**: Validate end-to-end latency and accuracy with PC inference
2. **Medium Term**: 
   - Port TFLite model to ESP32 using TensorFlow Lite Micro
   - Implement identical preprocessing on device (load scaler.pkl equivalent)
   - Optimize firmware for real-time sampling + inference cycle
3. **Long Term**:
   - Eliminate UDP transmission latency
   - Achieve true standalone edge device
   - Explore battery operation and power optimization
   - Consider sensor fusion or additional modalities if needed

The LABCA/gesture-cmd system represents a well-engineered, thoughtful approach to Edge AI gesture recognition that balances research rigor with practical deployment considerations.