# Edge AI Pipeline - Chronological Engineering Log

## Phase 0: Foundation Setup and General ML Exploration (July-August 2024)

### exp1: Initial Environment and Data Exploration
- **Date:** July-August 2024
- **Activities:**
  - Created Python virtual environment (.venv) with comprehensive ML stack
  - Explored `dataset.csv` (7.4MB) containing accelerometer data (acc_x, acc_y, acc_z) with timestamps
  - Developed basic data analysis tools:
    - `data_check.py`: Duplicate detection
    - `data.py`: Data loading with Polars
    - `data_graphs.py`: Distribution and correlation analysis
    - `min-max.py`: Extremum identification
    - `memory.py`: Memory usage tracking
    - `test.py`: Experimental validation
- **Key Insight:** This dataset appears unrelated to IMU gesture data (acc_y values ~10m/s² suggest vehicular motion), indicating early exploratory phase before focusing on gesture recognition

### exp2: IoT Telemetry and Feature Engineering
- **Date:** July 2024
- **Activities:**
  - Processed large-scale IoT telemetry datasets:
    - `iot_telemetry_data.csv` (619MB raw)
    - `environment_feature_engineered.csv` (543MB processed)
  - Developed feature engineering pipeline:
    - `dataset_exploration.py`: Initial data examination
    - `clean_data.py`: Data cleaning procedures
    - `data_normalization.py`: Normalization techniques
    - `categorical_encoding.py`: Encoding categorical variables
    - `feature_engineering.py`: Feature creation
    - `performance.py`: Model evaluation
- **Key Insight:** Transitioned from basic data exploration to applied ML on sensor telemetry data, establishing feature engineering workflows

### exp3: Classic ML Benchmarking
- **Date:** July-August 2024
- **Activities:**
  - Worked with standard ML datasets:
    - `train.csv`, `test.csv` (generic classification)
    - `Iris.csv` (Fisher's classic dataset)
  - Implemented train/test split workflows
  - Established baseline ML experimentation patterns
- **Key Insight:** Moved towards structured ML workflows with proper data partitioning, preparing for more sophisticated model development

## Phase 1: Model Optimization and EdgeAI Awareness (August-September 2024)

### exp4: Systematic Model Comparison and TinyNN Experiments
- **Date:** August 2024
- **Activities:**
  - **Iris Classification Benchmark:**
    - `RegularNN_Iris.keras`: Standard neural network (663KB)
    - `TinyNN_Iris.keras`: Compact/optimized neural network (28KB) - **first explicit EdgeAI model size awareness**
    - Generated standard train/test splits: `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`
    - Training history tracking: `train_accuracy.npy`, `train_loss.npy`, `val_accuracy.npy`, `val_loss.npy`
  - **Model Comparison Suite:**
    - `model_comparison.py`: Systematic architecture comparison
    - `All_Model_Comparison.csv`: Comprehensive results
    - `GridSearch_Results.csv`: Hyperparameter tuning outcomes
    - `Hyperparameter_Tuning_Summary.csv`: Tuning process documentation
    - `Evaluation_metrics.npy`: Stored metrics
  - **Experiment Tracking:**
    - `database.sqlite`: SQLite database for experiment metadata (10MB)
    - Comparative CSV files: suitability and performance analyses
    - Progressive task files (`task1.py` through `task7.py`)
- **Key Insight:** First explicit focus on model size constraints for deployment, establishing EdgeAI mindset with TinyNN_Iris.keras

### exp5: Model Pruning and Quantization Preparation
- **Date:** August-September 2024
- **Activities:**
  - Continued Iris dataset experiments with optimization focus
  - **Pruning Experiments:** `TinyNN_Iris_pruned.h5` and stripped variants
  - TFLite model exploration: `exp5/tflite_models/` with quantized, pruned, and float16 versions
- **Key Insight:** Advanced model optimization techniques for deployment - pruning and quantization pipeline establishment

### exp6: Telemetry Analysis and TFLite Micro Integration
- **Date:** September 2024
- **Activities:**
  - Telemetry data processing workflows:
    - `generate_test_data.py`: Synthetic test data creation
    - `check.py`: Data validation procedures
    - `convert_int8.py`: INT8 conversion tools
  - Embedded ML preparation:
    - Model data headers: `model_data.h`, `test_data.h`
    - TensorFlow Lite Micro integration: `exp6/tflite-micro/` with examples (hello_world, mnist_lstm, etc.)
- **Key Insight:** Shift towards embedded ML deployment readiness with TFLite Micro framework exploration

## Phase 2: Application-Specific Development (September-October 2024)

### exp7: Initial Gesture Recognition Modeling
- **Date:** September 2024
- **Activities:**
  - Gesture CNN model development:
    - `gesture_cnn_float.keras`: Floating-point baseline model
    - `gesture_cnn_int8.tflite`: Quantized deployment model
  - Task-based implementation: `exp7/task1.py` through `task5.py` (progressive feature development)
  - Performance analysis: `final_confusion_matrix.csv`
- **Key Insight:** First dedicated gesture recognition modeling effort, establishing 1D-CNN approach for temporal IMU data

### exp8: Hardware Deployment and Inference Profiling
- **Date:** September-October 2024
- **Activities:**
  - ESP32 implementation: `exp8/esp32/esp32.ino` - LDR-based laser classifier using TFLite Micro
  - Model data: `exp8/model_data.h`
  - Likely model generation: `exp8/esp8_model.py` (for laser detection)
  - Profiling focus: Real-time inference timing measurements
- **Key Insight:** Transition to hardware deployment and real-time inference profiling on ESP32, validating deployment feasibility

## Phase 3: Complete Portable System Development (Ongoing through October 2024)

### LABCA/gesture-cmd: End-to-End Gesture Recognition Wearable
- **Date:** Ongoing through October 2024
- **Activities:**

#### Firmware Development (`LABCA/gesture-cmd/firmware/src/`):
- `imu_reader.h`: MPU6050 I2C driver with register definitions (PWR_MGMT_1=0x6B, ACCEL_CONFIG=0x1C, GYRO_CONFIG=0x1B, ACCEL_XOUT_H=0x3B)
- `data_collection.cpp`: Serial data collection firmware (Phase 01) - CSV output via UART
- `main.cpp`: UDP IMU streaming firmware (Phase 03) - WiFi/UDP transmission to PC
- `window_buffer.h`: Circular buffer for sliding window processing (50 samples @ 50Hz = 1-second windows)

#### Server-side ML Pipeline (`LABCA/gesture-cmd/server/src/training/`):
- `gesture_logger.py`: Serial data collection and labeling system ('s'/'x' commands for window control)
- `preprocess.py`: 
  - Windowing strategy: 50 samples = 1-second windows
  - Train/validation/test split (70/15/15) with stratification
  - StandardScaler normalization (fit on training data only)
  - Artifact persistence: X_*.npy, y_*.npy, label_map.json, scaler.pkl
- `train_model.py`: 
  - Lightweight 1D-CNN architecture (~15K parameters target)
  - 3 Conv1D blocks with BatchNorm, MaxPooling, Dropout
  - GlobalAveragePooling1D + Dense classification head
  - Early stopping and learning rate scheduling
- `quantize_export.py`: 
  - INT8 post-training quantization with representative dataset
  - TFLite conversion and C header export for embedded deployment
  - Accuracy validation pre/post quantization

#### Inference and HCI (`LABCA/gesture-cmd/server/src/`):
- `inference_server.py`: UDP packet receiver → TFLite inference → MQTT publisher
- `hci_controller.py`: MQTT subscriber → keyboard shortcut translation (with uinput/Wayland support)

#### Documentation and Configuration:
- `Hardware_confs.md`: Key hardware decisions and experimental rationales
- `dataset.txt`: Collected gesture dataset in windowed CSV format
- `readings.txt`: Additional sensor readings and calibration data

#### Hardware Validation Platforms:
- `labca_accelerometer_test/`: Raw MPU6050 sensor verification and communication testing
- `first_dry_run/`: Initial MPU6050 + OLED display integration test
- `esp_test/`: ESP32 basic functionality testing
- `esp_test_oled_screen/`: ESP32 with display output for diagnostics

## Key Technical Evolution Patterns Observed

### 1. Progressive Specialization
- General ML exploration → IoT telemetry → Classic benchmarks → Model optimization → Application-specific gesture recognition → Complete wearable system

### 2. Hardware Standardization
- Initial sensor exploration → MPU6050 IMU standardization → ESP8266 → ESP32 consideration → Finalized ESP32-WROOM with MPU6050 at GPIO21/22

### 3. ML Model Maturation
- Standard datasets (Iris) → Size-aware experiments (TinyNN) → Application-specific architecture (1D-CNN for temporal data) → Deployment optimization (quantization, pruning)

### 4. Data Pipeline Rigor
- Basic CSV exploration → Windowing strategy establishment (50 samples = 1 sec) → Proper train/val/test splits → Standardization with scaler persistence → Artifact versioning

### 5. Deployment Consciousness
- Early model size awareness (TinyNN_Iris.keras at 28KB) → Quantization pipeline establishment → INT8 TFLite conversion → C header embedding for Arduino/ESP32

## Critical Decision Points Identified

1. **Sensor Selection:** MPU6050 chosen for 6-axis IMU (accelerometer + gyroscope) providing sufficient motion dynamics for gesture discrimination
2. **Communication Protocol:** I2C selected for simplicity and adequate bandwidth (50Hz × 6 channels = 300 values/sec)
3. **Sampling Rate:** 50Hz chosen as Nyquist-appropriate for human gesture frequencies (<25Hz) while balancing power and data throughput
4. **Window Size:** 50 samples = 1-second window captures full gesture temporal dynamics while maintaining reasonable latency
5. **Model Architecture:** 1D-CNN selected over dense/RNN for temporal feature extraction with parameter efficiency
6. **Quantization Strategy:** INT8 post-training quantization chosen for 4x model size reduction with minimal accuracy impact
7. **Gesture Set Refinement:** Initial 7-class system reduced to 6 classes after `fist_hold` showed insufficient separability from `idle`

This chronological log demonstrates a deliberate, iterative engineering process where each phase built upon lessons learned from previous experiments, culminating in a sophisticated EdgeAI wearable system.