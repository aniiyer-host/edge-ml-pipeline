# Edge AI Gesture Recognition Project - Technical History

## Chronological Project Log

### Phase 0: Initial Setup and Exploration (exp1)
**Timeline:** July-August 2024
**Focus:** Foundation setup, data exploration, basic ML concepts

**Key Activities:**
- Created virtual environment (.venv) with extensive Python dependencies including TensorFlow, Keras, Polars, scikit-learn
- Initial dataset exploration in `exp1/dataset.csv` (7.4MB) containing accelerometer data (acc_x, acc_y, acc_z) with timestamp and activity labels
- Basic data analysis scripts:
  - `data_check.py`: Checking for duplicates in dataset
  - `data.py`: Basic data loading with Polars
  - `data_graphs.py`: Histogram and correlation matrix exploration
  - `min-max.py`: Finding min/max values
  - `memory.py`: Memory usage tracking
  - `test.py`: Experimental validation
- **Observation:** This dataset appears to be from a different experiment (possibly vehicle/traffic data given acc_y values around 10m/s²), not the IMU gesture data that would come later

### Phase 1: IoT and Telemetry Experiments (exp2)
**Timeline:** July 2024
**Focus:** Environmental sensor data, feature engineering

**Key Activities:**
- Large-scale IoT telemetry dataset (`iot_telemetry_data.csv`: 619MB)
- Feature engineering pipeline:
  - `dataset_exploration.py`: Initial data examination
  - `clean_data.py`: Data cleaning procedures
  - `data_normalization.py`: Normalization techniques
  - `categorical_encoding.py`: Encoding categorical variables
  - `feature_engineering.py`: Creating new features
  - `performance.py`: Model performance evaluation
- **Environmental dataset:** `environment_feature_engineered.csv` (543MB) - processed telemetry data
- **Observation:** Shift from basic data exploration to applied ML on sensor telemetry data

### Phase 2: Classic ML Benchmarking (exp3)
**Timeline:** July-August 2024
**Focus:** Establishing ML baselines with standard datasets

**Key Activities:**
- Standard ML datasets: `train.csv`, `test.csv`, `Iris.csv`
- Classification experiments with train/test splits
- **Observation:** Moving towards structured ML workflows with proper train/validation/test splits

### Phase 3: Model Comparison and Optimization (exp4)
**Timeline:** August 2024
**Focus:** Systematic model comparison, hyperparameter tuning, neural network optimization

**Key Activities:**
- **Iris Classification Benchmark:**
  - `Iris.csv`: Classic Fisher's Iris dataset
  - `RegularNN_Iris.keras`: Standard neural network (663KB)
  - `TinyNN_Iris.keras`: Compact/optimized neural network (28KB) - early EdgeAI experimentation
  - `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`: Standard train/test splits
  - Training history files: `train_accuracy.npy`, `train_loss.npy`, `val_accuracy.npy`, `val_loss.npy`
- **Model Comparison Suite:**
  - `model_comparison.py`: Systematic comparison of different architectures
  - `All_Model_Comparison.csv`: Results of model comparisons
  - `GridSearch_Results.csv`: Hyperparameter tuning results
  - `Hyperparameter_Tuning_Summary.csv`: Summary of tuning efforts
  - `Evaluation_metrics.npy`: Stored evaluation metrics
- **Support Files:**
  - `database.sqlite`: SQLite database for experiment tracking (10MB)
  - Various CSV files for suitability and performance comparisons
  - Task files (`task1.py` through `task7.py`): Progressive implementation steps
- **Observation:** First explicit EdgeAI experimentation with TinyNN_Iris.keras showing awareness of model size constraints for deployment

### Phase 4: ROS and Robotics Integration (exp5)
**Timeline:** August-September 2024
**Focus:** Robot Operating System (ROS) integration, pruning experiments

**Key Activities:**
- Continued Iris dataset experiments with model optimization
- **Pruning Focus:** `TinyNN_Iris_pruned.h5` and stripped variants
- TFLite model exploration: `exp5/tflite_models/` directory with quantized, pruned, and float16 variants
- **Observation:** Explicit model optimization for deployment - pruning and quantization experiments

### Phase 5: Telemetry Analysis (exp6)
**Timeline:** September 2024
**Focus:** Telemetry data processing, test data generation

**Key Activities:**
- Test data generation: `generate_test_data.py`
- Data validation: `check.py`
- INT8 conversion: `convert_int8.py`
- Model data headers: `model_data.h`, `test_data.h`
- TensorFlow Lite Micro integration: `exp6/tflite-micro/` with examples
- **Observation:** Shift towards embedded ML deployment preparation with TFLite Micro

### Phase 6: Gesture Recognition Foundation (exp7)
**Timeline:** September 2024
**Focus:** Initial gesture recognition model development

**Key Activities:**
- Gesture CNN models: `gesture_cnn_float.keras`, `gesture_cnn_int8.tflite`
- Task-based implementation: `exp7/task1.py` through `task5.py`
- Confusion matrix analysis: `final_confusion_matrix.csv`
- **Observation:** First dedicated gesture recognition modeling effort

### Phase 7: Laser Classification Profiling (exp8)
**Timeline:** September-October 2024
**Focus:** ESP32 deployment, inference profiling

**Key Activities:**
- ESP32 implementation: `exp8/esp32/esp32.ino` - LDR-based laser classifier using TFLite Micro
- Model data: `exp8/model_data.h`
- Profiling focus: `exp8/esp8_model.py` (likely model generation for laser detection)
- **Observation:** Transition to hardware deployment and real-time inference profiling on ESP32

### Phase 8: LABCA Gesture Command System (Main Project)
**Timeline:** Ongoing through October 2024
**Focus:** Complete end-to-end gesture recognition wearable system

**Key Activities:**
- **Firmware Development** (`LABCA/gesture-cmd/firmware/src/`):
  - `imu_reader.h`: MPU6050 I2C driver with register definitions
  - `data_collection.cpp`: Serial data collection firmware (Phase 01)
  - `main.cpp`: UDP IMU streaming firmware (Phase 03)
  - `window_buffer.h`: Circular buffer for sliding window processing
- **Server-side ML Pipeline** (`LABCA/gesture-cmd/server/src/training/`):
  - `gesture_logger.py`: Serial data collection and labeling
  - `preprocess.py`: Windowing, normalization, train/val/test split
  - `train_model.py`: 1D-CNN model training (~15K parameters target)
  - `quantize_export.py`: INT8 quantization and C header export
- **Inference and HCI** (`LABCA/gesture-cmd/server/src/`):
  - `inference_server.py`: UDP→TFLite→MQTT inference bridge
  - `hci_controller.py`: MQTT→keyboard shortcuts translation
- **Documentation and Config:**
  - `Hardware_confs.md`: Key hardware decisions and rationales
  - `dataset.txt`: Collected gesture dataset
  - `readings.txt`: Additional sensor readings
- **Hardware Test Platforms:**
  - `labca_accelerometer_test/`: Raw MPU6050 sensor verification
  - `first_dry_run/`: Initial MPU6050 + OLED test
  - `esp_test/`: ESP32 basic testing
  - `esp_test_oled_screen/`: ESP32 with display output

## Key Evolution Observations

1. **Progression from General ML to Specific Application:**
   - Started with data exploration (exp1-exp4)
   - Moved through benchmarking and optimization (exp4-exp6)
   - Focused on gesture recognition (exp7-exp8)
   - Culminated in complete wearable system (LABCA/gesture-cmd)

2. **Hardware Evolution:**
   - Initial exploration with various sensors (exp1-exp2)
   - Standardized on MPU6050 IMU for motion sensing
   - Progressed from ESP8266 to ESP32 consideration
   - Finalized ESP32-WROOM with MPU6050 at GPIO21/22

3. **ML Model Evolution:**
   - Started with standard datasets (Iris)
   - Progressed to TinyNN experiments (size-aware models)
   - Developed specialized 1D-CNN for temporal gesture data
   - Implemented quantization pipeline for deployment

4. **Data Pipeline Maturity:**
   - Early CSV exploration
   - Developed windowing strategy (50 samples = 1 second)
   - Established preprocessing pipeline with proper train/val/test splits
   -Implemented standardization with scaler persistence