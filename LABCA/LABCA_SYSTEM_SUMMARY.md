# LABCA Gesture Recognition System - Summary

## System Overview
LABCA/gesture-cmd is a complete end-to-end gesture recognition wearable system that:
- Uses ESP32/ESP8266 + MPU6050 IMU to capture hand motions
- Processes data through a rigorous machine learning pipeline
- Translates recognized gestures into keyboard shortcuts on a PC
- Designed for eventual deployment to Edge AI (on-device inference)

## Hardware Configuration (Final)

### Core Components
- **Microcontroller**: ESP32-WROOM DevKit
- **IMU Sensor**: MPU6050 (GY-521 breakout) 
- **Communication**: I²C Bus (SDA = GPIO21, SCL = GPIO22)
- **Power**: USB 5V (development)
- **Optional Display**: SSD1306 OLED (I²C address 0x3C)

### Sensor Configuration
- **Accelerometer Range**: ±2g (scale: 16384 LSB/g)
- **Gyroscope Range**: ±1000°/s (scale: 32.8 LSB/(°/s)) 
  *Selected to prevent saturation during wrist rotation*
- **Sampling Frequency**: 50 Hz
- **Window Size**: 50 samples = 1.0 second duration
- **Data Channels**: 6-axis (ax, ay, az, gx, gy, gz)

### Wiring
```
MPU6050        ESP32
VCC ---------- 3V3
GND ---------- GND
SCL ---------- GPIO22
SDA ---------- GPIO21
AD0 ---------- GND (fixes I²C address to 0x68)
INT ---------- NC
```

## Firmware Architecture (3 Phases)

### Phase 01: Data Collection Firmware (`data_collection.cpp`)
- **Purpose**: Serial logging of IMU data for dataset creation
- **Features**: 
  - Direct I2C communication via Wire.h
  - MPU6050 init: wake (PWR_MGMT_1=0x00), ±2g accel, ±250°/s gyro
  - 50Hz sampling (20ms interval) with 50-sample windows
  - Serial commands: 's'=start window, 'x'=discard window
  - Output: CSV lines `timestamp_ms,ax,ay,az,gx,gy,gz`
  - Auto-stop after 50 samples

### IMU Reader Library (`imu_reader.h`)
- **Purpose**: Reusable, lightweight MPU6050 driver
- **Features**:
  - Register-level access (no external libraries)
  - Register definitions: PWR_MGMT_1=0x6B, ACCEL_CONFIG=0x1C, GYRO_CONFIG=0x1B, ACCEL_XOUT_H=0x3B
  - Functions: initIMU(), readIMU()
  - Scaling: ACCEL_SCALE=16384.0f, GYRO_SCALE=131.0f

### Phase 03: UDP IMU Streaming Firmware (`main.cpp`)
- **Purpose**: Wireless transmission to PC for real-time inference
- **Features**:
  - WiFi connection to LAN
  - UDP packet transmission (1208 bytes):
    - [0-3]: uint32 sequence number (loss detection)
    - [4-7]: uint32 timestamp_ms
    - [8-1207]: float32[50][6] window data
  - Uses imu_reader.h + window_buffer.h
  - Continuous 50Hz sampling → window fill → transmit
  - NO on-device ML (PC-based inference for easier debugging)

## Machine Learning Pipeline

### Data Flow: Raw CSV → Deployable Model
1. **Data Collection**: `gesture_logger.py` captures labeled CSV from serial
   - Format: `label,sample_id,timestamp_ms,sample,ax,ay,az,gx,gy,gz`
   - Window-based labeling: 's' to start, auto-saves after 50 samples
2. **Preprocessing**: `preprocess.py` creates model-ready tensors
   - Window validation: exactly 50 samples/window, label-consistent
   - Train/val/test split: stratified 70/15/15
   - StandardScaler: fitted ONLY on training data (critical anti-leakage)
   - Output: X_{train,val,test}.npy, y_{train,val,test}.npy
3. **Model Training**: `train_model.py` builds lightweight 1D-CNN
   - Architecture: 
     ```
     Input: (50,6)
     → Conv1D(32,k=5)+BN+MaxPool+Dropout(0.3)
     → Conv1D(64,k=3)+BN+MaxPool+Dropout(0.3)
     → Conv1D(64,k=3)+BN+Dropout(0.3)
     → GlobalAvgPool1D
     → Dense(64,ReLU)+Dropout(0.3)
     → Dense(n_classes,Softmax)
     ```
   - ~24K parameters (~15KB float32)
   - Adam(lr=1e-3), sparse categorical crossentropy
   - Early stopping, LR scheduling, checkpointing
4. **Quantization**: `quantize_export.py` creates deployment model
   - INT8 post-training quantization
   - Representative dataset: 200 training samples for calibration
   - Results: 
     - Size: 107,044 → 41,448 bytes (61.3% reduction)
     - Accuracy: 99.39% → 99.39% (0% drop)
   - Outputs:
     - `gesture_model_int8.tflite` (deployable)
     - `gesture_model.h` (C header for Arduino embedding)
     - `label_map.json`, `scaler.pkl` (preprocessing artifacts)

## Real-Time Deployment Architecture

### Inference Pipeline
```
[ESP32 Firmware] 
        │ Samples MPU6050 @ 50Hz
        │ 50 samples = 1-second window
        ▼
[UDP Packet: 1208 bytes] ───▶ [PC: inference_server.py]
        │ [seq][timestamp][float32[50][6]]
        ▼
[Preprocessing] ───────────▶ [Apply SAME StandardScaler as training]
        │ (window - μ_train) / σ_train per channel
        ▼
[TFLite INT8 Inference] ───▶ [Quantize→Compute→Dequantize→Softmax]
        │
[MQTT Publish] ───────────▶ [PC: hci_controller.py]
        │ {"gesture":"wave_right","confidence":0.96}
        ▼
[Keyboard Shortcut] ─────▶ [Host OS Action]
        │ e.g., Ctrl+Right → Next desktop
        ▼
[User Perceives Response]
```

### Key Anti-Leakage Measures
- **Window Integrity**: Never split samples from same window across train/val/test
- **Scalar Fitting**: StandardScaler fitted ONLY on training data
- **Identical Preprocessing**: Same μ, σ used at inference as in training
- **Stratified Splits**: Maintains class distribution during splitting
- **No Sample Editing**: Corrupted windows rejected entirely (never repaired)
- **Quantization Calibration**: Uses training data only for INT8 range determination

## Gesture Set (Final 6 Classes)
1. **idle**: Resting hand position
2. **wave_left**: Horizontal left-to-right wave
3. **wave_right**: Horizontal right-to-left wave
4. **flick_up**: Vertical upward flick
5. **flick_down**: Vertical downward flick
6. **wrist_rotate**: Rotational wrist motion
   *Note: fist_hold was removed due to overlap with idle*

## Technical Decisions & Rationales

| Decision | Reason |
|----------|--------|
| **Direct Register Access** | Avoided library compile issues; minimal dependencies |
| **UART Data Collection** | Simplicity & debugging visibility during dataset creation |
| **UDP Transmission** | Low overhead, simplicity; app-level reliability sufficient |
| **PC-Based Inference** | Easier debugging/iteration; validates pipeline pre-port |
| **1D-CNN Architecture** | Extracts temporal features efficiently; parameter constraints |
| **Progressive Kernels (5→3→3)** | Capture multi-scale temporal patterns |
| **GlobalAvgPool1D** | Dramatically reduces params before Dense layers (key to size target) |
| **INT8 Quantization** | 4x size reduction with negligible accuracy impact |
| **50Hz/50-sample Window** | Nyquist-appropriate for human motion; balances latency/completeness |
| **±1000°/s Gyro Range** | Prevents saturation during wrist rotation (tested to ~±822°/s) |
| **Window-Based Labeling** | Uniform input size; allows natural gesture timing variability |

## Strengths of This Approach
1. **Reproducibility**: Rigorous anti-leakage data pipeline
2. **Modularity**: Clear separation (firmware, server, interface)
3. **Deployment Awareness**: Architecture designed with MCU constraints
4. **Iterative Development**: PC-based inference enables rapid prototyping
5. **User-Centered Design**: Natural timing variation encouraged in collection
6. **Transparency**: Direct register access, documented decisions

## Path to True Edge Deployment
Current PC-based inference validates the complete pipeline. Future work:
1. Port TFLite model to ESP32 using TensorFlow Lite Micro
2. Implement identical preprocessing on device (load scaler.pkl)
3. Optimize firmware for real-time sampling → inference cycle
4. Eliminate UDP transmission latency
5. Achieve true standalone edge device operation
6. Explore battery operation and power optimization

The LABCA/gesture-cmd system represents a well-engineered, thoughtful approach to Edge AI gesture recognition that balances research rigor with practical deployment considerations, establishing a complete pipeline ready for eventual on-device migration.