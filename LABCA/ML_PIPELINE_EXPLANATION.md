# Edge AI Gesture Recognition Project - Machine Learning Pipeline Explanation

## End-to-End ML Pipeline Overview

The gesture recognition system follows a sophisticated machine learning pipeline designed for Edge AI deployment, with careful attention to preventing data leakage and ensuring reproducible results. The pipeline processes raw IMU data through several stages to produce a deployable quantized model.

## Pipeline Stages

### Stage 1: Raw Data Collection
**Input:** Serial data from ESP32/MPU6050 via UART
**Format:** CSV lines: `timestamp_ms,ax,ay,az,gx,gy,gz`
**Process:**
- ESP32 samples MPU6050 at 50Hz (20ms interval)
- Data collection firmware buffers samples until window command received
- Serial logger captures labeled data windows
- Each window tagged with gesture label and sample_id

**Key Design:** Fixed 50-sample windows (1 second duration) ensure uniform input size for model training.

### Stage 2: Data Validation and Window Integrity
**Input:** Raw CSV data file
**Validation Checks:**
- Exactly 50 samples per window
- Correct label consistency within window
- Numeric sensor values (no NaN or Inf)
- Chronological sample ordering (sample 0-49)
- No missing values
- No duplicate rows
- Window boundaries marked by START/END flags

**Output:** Cleaned dataset with validated windows only
**Rejection Policy:** Entire window rejected if any validation fails - no sample-level editing to preserve data integrity

### Stage 3: Window Extraction and Structuring
**Input:** Validated CSV dataset
**Process:**
- Groups data by (`label`, `sample_id`) 
- Each group forms one 50×6 tensor (50 timesteps, 6 features: ax,ay,az,gx,gy,gz)
- Discards incomplete windows (<50 samples)
- Returns:
  - X: numpy array of shape (N_windows, 50, 6)
  - y_strings: list of gesture labels per window

**Critical Design Feature:** Window-level grouping prevents data leakage - never splits samples from same window across train/val/test sets

### Stage 4: Label Encoding
**Input:** List of string labels per window
**Process:**
- Uses sklearn LabelEncoder to map gesture names to integers
- Creates bidirectional mapping: label_map.json (name→index and index→name)
- Transforms y_strings to y_int (numpy array of integers)

**Output:** 
- X: (N_windows, 50, 6) float32
- y: (N_windows,) int32
- label_map.json: persistent mapping for inference

### Stage 5: Train/Validation/Test Split
**Input:** Features (X) and labels (y)
**Process:**
- Stratified split maintains class distribution across sets
- First split: 85% (temp) for train+val, 15% for test
- Second split: 70% train, 15% val from temp set (results in 70/15/15 final split)
- Uses random_state=42 for reproducibility
- Stratification ensures similar gesture proportions in each split

**Critical Design Feature:** Stratified splitting prevents class imbalance that could bias model training or evaluation

### Stage 6: Feature Normalization (StandardScaler)
**Input:** Training, validation, and test sets
**Process:**
- **Fit ONLY on training data:** Computes mean and std per feature channel from X_train
- **Transform all sets:** Applies same scaling to X_train, X_val, X_test
- Formula: X_scaled = (X - mean) / std
- Applied per-channel: each of 6 features normalized independently
- Reshapes to (N*50, 6) for computing statistics, then back to (N, 50, 6)

**Critical Design Feature:** Fitting scaler ONLY on training data prevents information leakage from validation/test sets into training process

### Stage 7: Model Training (1D-CNN)
**Input:** Normalized training data (X_train, y_train)
**Process:**
- Builds lightweight 1D Convolutional Neural Network
- Architecture:
  - Input: (50, 6) - 50 timesteps, 6 IMU channels
  - Conv1D(32, kernel_size=5) + BatchNorm + MaxPooling1D + Dropout(0.3)
  - Conv1D(64, kernel_size=3) + BatchNorm + MaxPooling1D + Dropout(0.3)  
  - Conv1D(64, kernel_size=3) + BatchNorm + Dropout(0.3)
  - GlobalAveragePooling1D
  - Dense(64, activation='relu') + Dropout(0.3)
  - Dense(n_classes, activation='softmax')
- Compiled with Adam optimizer (lr=1e-3), sparse categorical crossentropy loss
- Uses callbacks:
  - EarlyStopping (patience=15, restore_best_weights)
  - ReduceLROnPlateau (factor=0.5, patience=8)
  - ModelCheckpoint (save best model)
- Trains for up to 80 batches with batch size 32

**Output:** Trained Keras model (`gesture_model.keras`)
**Validation:** Monitored val_accuracy to prevent overfitting

### Stage 8: Model Evaluation
**Input:** Trained model and test set (X_test, y_test)
**Process:**
- Generates predictions on test set
- Computes accuracy, precision, recall, F1-score per class
- Creates confusion matrix to identify misclassification patterns
- Compares against target accuracy (>92%)
- If target not met, recommends:
  - Collecting more training samples (400+ per class)
  - Checking sensor mounting and stability
  - Verifying preprocessing consistency
  - Tuning hyperparameters

**Output:** Evaluation metrics and confusion matrix

### Stage 9: INT8 Quantization
**Input:** Trained Keras model and training data (for calibration)
**Process:**
- Uses TensorFlow Lite Converter with optimization flags:
  - `tf.lite.Optimize.DEFAULT`
  - Representative dataset (200 samples from X_train) for calibration
  - Sets supported ops to TFLITE_BUILTINS_INT8
  - Sets inference input/output type to int8
- Converts model to TFLite flatbuffer with full integer quantization
- Weights, activations, and intermediate tensors all quantized to int8

**Calibration:** Representative dataset determines dynamic range of each layer for optimal quantization

### Stage 10: Quantized Model Evaluation
**Input:** INT8 TFLite model and test set
**Process:**
- Runs TFLite interpreter on test set
- Handles quantization/dequantization:
  - Input: float32 → int8 using layer-specific scale/zero_point
  - Output: int8 → float32 using layer-specific scale/zero_point
- Computes accuracy of quantized model
- Compares against float32 baseline to measure accuracy impact
- Generates quantization report showing size and accuracy metrics

**Output:** 
- `gesture_model_int8.tflite` - quantized model ready for deployment
- `gesture_model.h` - C header array for Arduino/ESP32 embedding
- `quantization_report.txt` - detailed comparison

### Stage 11: Deployment Preparation
**Input:** Quantized TFLite model and metadata
**Process:**
- Packages model with:
  - label_map.json (gesture ↔ index mapping)
  - scaler.pkl (for identical preprocessing on device)
  - Model architecture details
- Creates inference server that:
  - Receives UDP packets from ESP32
  - Applies same standardization as training
  - Runs TFLite inference
  - Publishes results via MQTT
- Deploys HCI controller that:
  - Subscribes to MQTT gesture topics
  - Maps gestures to keyboard shortcuts
  - Executes shortcuts on host PC

## Critical Anti-Leakage Measures

1. **Window Integrity:** Never splits samples from same window across dataset splits
2. **Scalar Fitting:** StandardScaler fitted ONLY on training data
3. ** Stratified Splits:** Maintains class distribution to prevent bias
4. **Quantization Calibration:** Uses training data only for determining int8 ranges
5. **Reproducibility:** Fixed random seeds and deterministic processing

## Deployment Artifacts

The pipeline produces these deployable artifacts:
1. `gesture_model_int8.tflite` - Quantized model (~15KB)
2. `gesture_model.h` - C header for direct ESP32 embedding
3. `label_map.json` - Gesture class mapping
4. `scaler.pkl` - Identical preprocessing for on-device use
5. `quantization_report.txt` - Validation of quantization impact

## Inference Pipeline on Device

For eventual ESP32 deployment:
1. ESP32 samples MPU6050 at 50Hz into 50-sample window
2. Apply same standardization: (x - mean_train) / std_train
3. Quantize to int8 using model's input scale/zero_point
4. Run TFLite Micro inference
5. Dequantize output using model's output scale/zero_point  
6. Apply softmax to get class probabilities
7. Select class with highest probability (if above confidence threshold)

This ensures identical processing between training and deployment environments.