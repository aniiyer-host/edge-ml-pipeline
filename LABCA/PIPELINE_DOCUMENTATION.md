# TinyML Gesture Recognition Pipeline Documentation

## Project Overview and Goals

This document describes the complete TinyML gesture recognition pipeline for an ESP32-WROOM microcontroller interfacing with an MPU6050 IMU sensor. The pipeline covers end-to-end processing from raw sensor data collection through model training, evaluation, quantization, and preparation for embedded deployment.

**Hardware Configuration:**
- Microcontroller: ESP32-WROOM
- IMU Sensor: MPU6050 via I2C
- I2C SDA: GPIO 21
- I2C SCL: GPIO 22
- MPU6050 I2C Address: 0x68
- Accelerometer Range: ±2g (16384 LSB/g)
- Gyroscope Range: ±1000°/s (32.8 LSB/(°/s))
- Sampling Rate: 50 Hz (20 ms interval)
- Window Length: 50 samples (1 second)
- Input Channels: 6 (ax, ay, az, gx, gy, gz)
- Acceleration Units: g
- Gyroscope Units: degrees/second

**Gesture Classes (6 total):**
- 0: idle
- 1: wave_left
- 2: wave_right
- 3: flick_up
- 4: flick_down
- 5: wrist_rotate

## Hardware and Data Specification

The sensor configuration, channel ordering, units, sample rate, and window length must remain consistent between data collection, training, and inference to ensure:
1. Feature distribution alignment between training and deployment
2. Correct interpretation of sensor values (g and °/s units)
3. Temporal coherence of windowed samples (1-second gestures)
4. Identical preprocessing steps (normalization, quantization)

## Dataset Audit and Cleaning

### Raw Dataset Findings (LABCA/dataset.txt)
- Total lines: 4,319
- Complete sensor windows: 65 (3,250 sensor rows)
- Embedded CSV header lines: 65
- Comment lines (starting with #): 846
- Free-text notes: 3
- Window distribution per class:
  - idle: 10 windows
  - wave_left: 10 windows
  - wave_right: 10 windows
  - flick_up: 10 windows
  - flick_down: 10 windows
  - wrist_rotate: 15 windows
- No malformed numeric rows identified
- No out-of-range samples identified during audit
- Two az values approaching accelerometer saturation in wrist_rotate windows 76 and 77

### Cleaning Operations
The raw dataset was processed to create `gesture-cmd/server/data/gesture_dataset_clean.csv` with the following operations:
1. Removal of all comment lines (lines starting with #)
2. Removal of embedded CSV header lines (duplicate "label,sample_id,sample,timestamp_ms,ax,ay,az,gx,gy,gz" lines)
3. Preservation of all sensor measurements exactly as recorded
4. No bias correction, filtering, or alteration of sensor values
5. Retention of suspicious az readings in wrist_rotate windows 76 and 77 for investigation

The cleaned dataset schema is:
`label,sample_id,sample,timestamp_ms,ax,ay,az,gx,gy,gz`

### Verification of Clean Dataset
- Classes present: exactly the six expected gestures
- Sample count verification: 65 complete windows
- No anomalous labels or metadata in cleaned data

## Data Schema and Gesture Labels

### Fixed Class Mapping (Interface Contract)
| Class Index | Gesture      | Verified Source |
| ----------: | ------------ | --------------- |
|           0 | idle         | label_map.json  |
|           1 | wave_left    | label_map.json  |
|           2 | wave_right   | label_map.json  |
|           3 | flick_up     | label_map.json  |
|           4 | flick_down   | label_map.json  |
|           5 | wrist_rotate | label_map.json  |

This mapping is stored in `gesture-cmd/server/edgeai_v2/data/label_map.json` and must remain constant across training, quantization, and firmware implementation.

## Windowing, Splitting, and Normalization

### Windowing Strategy
- Groups complete windows by (`label`, `sample_id`) pairs
- Validates each window has exactly 50 samples (WINDOW_SIZE)
- Verifies sample indices are sequential 0-49
- Rejects incomplete windows or malformed sample sequences
- Each window represents a 1-second gesture instance

### Splitting Strategy
- **Method**: Deterministic stratified 60%/20%/20% split at window level
- **Implementation**: 
  - Split performed independently within each gesture class
  - No individual timesteps from a window can leak across splits (prevents data leakage)
  - Random seed fixed at 42 for reproducibility
  - Shuffling applied after split assignment
- **Verification**: Split membership saved in `split_manifest.json`

### Normalization Procedure
- **Fit**: Compute mean and standard deviation **only on training windows**
- **Transformation**: 
  - Reshape data to (N_windows * 50, 6) features
  - Apply z-score normalization: (x - mean) / std
  - Replace near-zero std values (≤ 1e-8) with 1.0 to prevent division by zero
- **Persistence**: 
  - Saved statistics in `scaler_stats.npz` (mean and std arrays)
  - Applied identically to validation and test sets
  - Must be replicated exactly in embedded firmware

### Verified Statistics from Training
```
Mean: [-0.04022  0.04036  0.92736 -2.39706  2.13902 -0.51755]
Std:  [0.45261  0.44438  0.30295 86.07196 85.28176 64.77739]
```

*Note: These values are specific to the cleaned dataset and must not be reused if data collection changes.*

## Model Architecture and Training

### Architecture Verification
Source: `LABCA/train_edgeai.py` (build_model function)

1. **Input**: Shape (50, 6) - 50 timesteps, 6 IMU features
2. **Conv1D**: 32 filters, kernel size 5, same padding, ReLU
3. **Batch Normalization**
4. **Max Pooling**: Pool size 2
5. **Dropout**: 0.3
6. **Conv1D**: 64 filters, kernel size 3, same padding, ReLU
7. **Batch Normalization**
8. **Max Pooling**: Pool size 2
9. **Dropout**: 0.3
10. **Conv1D**: 64 filters, kernel size 3, same padding, ReLU
11. **Batch Normalization**
12. **Global Average Pooling** (replaces flattening to reduce parameters)
13. **Dense**: 64 units, ReLU
14. **Dropout**: 0.3
15. **Output**: Dense with 6 units, softmax activation

### Training Configuration Verified
- **Optimizer**: Adam with learning rate 1e-3
- **Loss**: sparse_categorical_crossentropy
- **Metrics**: accuracy
- **Batch Size**: 8 windows
- **Maximum Epochs**: 120
- **Callbacks**:
  - EarlyStopping (monitor=val_loss, patience=20, restore_best_weights=True)
  - ReduceLROnPlateau (monitor=val_loss, factor=0.5, patience=8, min_lr=1e-5)
  - ModelCheckpoint (save best model by val_loss)
- **Seeds**: Fixed at 42 for Python, NumPy, and TensorFlow
- **Parameter Count**: 24,742 trainable parameters
- **Estimated Float32 Size**: ~96 KB (24,742 × 4 bytes)

### Model Artifacts
- `gesture_model.keras`: Final trained model (after EarlyStopping restore)
- `best_gesture_model.keras`: Checkpoint from best validation epoch
- Both files are identical due to restore_best_weights=True

## Evaluation Methodology and Limitations

### Test Results Verification
Source: `LABCA/gesture-cmd/server/edgeai_v2/evaluation_report.txt`

```
TensorFlow: 2.22.0-rc0
Seed: 42
Window count: 65
Parameter count: 24742
Test accuracy: 1.000000
Test macro F1: 1.000000

Classification report:
              precision    recall  f1-score   support
    idle       1.00      1.00      1.00         2
   wave_left       1.00      1.00      1.00         2
  wave_right       1.00      1.00      1.00         2
    flick_up       1.00      1.00      1.00         2
  flick_down       1.00      1.00      1.00         2
wrist_rotate       1.00      1.00      1.00         3

accuracy                           1.00        13
macro avg       1.00      1.00      1.00        13
weighted avg       1.00      1.00      1.00        13

Confusion matrix (rows=true, columns=predicted):
[[2 0 0 0 0 0]
 [0 2 0 0 0 0]
 [0 0 2 0 0 0]
 [0 0 0 2 0 0]
 [0 0 0 0 2 0]
 [0 0 0 0 0 3]]

Label order:
{
  "idle": 0,
  "wave_left": 1,
  "wave_right": 2,
  "flick_up": 3,
  "flick_down": 4,
  "wrist_rotate": 5
}
```

### Critical Limitations (Must Be Disclosed)
1. **Extremely Small Test Set**: Only 13 windows total (2-3 per class)
2. **Perfect Score Caveat**: 100% accuracy on this split does not demonstrate real-world robustness
3. **Session Correlation**: All windows collected in single session; temporal correlations may exist
4. **Single Participant**: Data represents one user's gesture variations
5. **Limited Environmental Variability**: No testing across different orientations, lighting, or motion contexts
6. **Tiny Pilot Dataset**: 65 total windows insufficient for generalization estimates

**The evaluation confirms the pipeline can achieve perfect separation on this specific split, but results should be interpreted as proof-of-concept only.**

## TFLite Conversion and Quantization

### Conversion Results Verification
Source: `LABCA/gesture-cmd/server/edgeai_v2/quantization_report.json`

```
{
  "tensorflow_version": "2.22.0-rc0",
  "label_map": {...},
  "keras_test_accuracy": 1.0,
  "keras_test_macro_f1": 1.0,
  "float32_tflite_test_accuracy": 1.0,
  "float32_tflite_macro_f1": 1.0,
  "keras_float32_max_abs_output_difference": 5.960464477539063e-08,
  "int8_tflite_test_accuracy": 1.0,
  "int8_tflite_macro_f1": 1.0,
  "float32_tflite_size_bytes": 109284,
  "int8_tflite_size_bytes": 43056,
  "int8_input": {
    "shape": [1, 50, 6],
    "dtype": "<class 'numpy.int8'>",
    "scale": 0.050871994346380234,
    "zero_point": -19
  },
  "int8_output": {
    "shape": [1, 6],
    "dtype": "<class 'numpy.int8'>",
    "scale": 0.00390625,
    "zero_point": -128
  }
}
```

### Quantization Procedure
1. **Float32 Baseline**: Direct TFLite conversion without quantization
2. **Full INT8 Quantization**:
   - Used all 39 training windows as calibration dataset
   - Converter optimizations: [tf.lite.Optimize.DEFAULT]
   - Target specs: [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
   - Force INT8 for both input and output tensors
3. **Accuracy Preservation**: 0% drop from Keras to INT8 TFLite on test set
4. **Output Equivalence**: Maximum absolute difference between Keras and float32 TFLite: 5.96e-08 (negligible)

### Quantization Formulas for Firmware
**Input Quantification** (float32 → int8):
```
q = round(x_normalized / input_scale + input_zero_point)
q = clamp(q, -128, 127)  // int8 range
```

**Output Dequantification** (int8 → float32):
```
x = (q - output_zero_point) * output_scale
```

**Classification**: Use argmax over six INT8 output values valid because:
- All output values share identical positive quantization scale (0.00390625)
- Identical zero point (-128)
- Therefore, ordering is preserved in quantized domain

### Operator Verification
The INT8 TFLite model contains these operators:
- EXPAND_DIMS
- CONV_2D  (Note: Conv1D operations converted to Conv2D with reshaping)
- MUL
- ADD
- RESHAPE
- MAX_POOL_2D
- MEAN (equivalent to GlobalAveragePooling1D)
- FULLY_CONNECTED
- SOFTMAX

*Note: Firmware must confirm operator support in TensorFlow Lite Micro for ESP32.*

## Artifact Inventory

### Models Directory (`gesture-cmd/server/edgeai_v2/models/`)
- `gesture_model.keras` (366,585 bytes): Final TensorFlow/Keras model
- `best_gesture_model.keras` (366,585 bytes): Identical to above (checkpoint)
- `gesture_model_float32.tflite` (109,284 bytes): Baseline float32 TFLite
- `gesture_model_int8.tflite` (43,056 bytes): Quantized INT8 TFLite (~60% size reduction)

### Data Directory (`gesture-cmd/server/edgeai_v2/data/`)
- `label_map.json` (109 bytes): Class-to-index mapping
- `scaler_stats.npz` (548 bytes): Normalization statistics (mean, std arrays)
- `split_manifest.json` (4,141 bytes): Deterministic split membership
- `splits_normalized.npz` (65,822 bytes): Pre-processed train/val/test arrays

### Documentation and Reports
- `evaluation_report.txt` (950 bytes): Test set evaluation results
- `training_curves.png` (56,623 bytes): Loss and accuracy plots
- `quantization_report.json` (836 bytes): Detailed quantization metrics
- `PIPELINE_DOCUMENTATION.md` (this file): Complete pipeline documentation

### ESP32 Firmware Directory (`gesture-cmd/server/edgeai_v2/esp32/`)
- `gesture_model_int8.h` (to be generated): C header array for embedded model

*Note: The textual size of the generated C header will be substantially larger than the raw model size due to hex representation overhead.*

## Embedded Deployment Status and Next Steps

### Current Status Verified
1. ✓ Model training completed on cleaned dataset
2. ✓ Float32 TFLite conversion verified accuracy parity
3. ✓ Full INT8 quantization completed with representative dataset
4. ✓ Quantization report shows 0% accuracy drop on test set
5. ✓ Generated INT8 TFLite model: 43,056 bytes
6. ✓ Normalization statistics persisted for firmware use
7. ✓ Label mapping exported as stable interface contract
8. ✗ Embedded inference on ESP32 **NOT YET IMPLEMENTED or TESTED**

### Firmware Preparation Status
- Selected board in Arduino IDE: `ESP32 Dev Module`
- Installed libraries:
  - `TensorFlowLite_ESP32` version 1.0.0
  - `Chirale_TensorFlowLite` version 2.0.0
- Note: Chirale library uses `Chirale_TensorFlowLite.h`, `AllOpsResolver`, `MicroInterpreter`
- Warning: Chirale Hello World example tested on Arduino Nano 33 BLE, **not ESP32**
- ESP32 compatibility, operator support, and tensor allocation **still to be verified**

### Required Next Steps for Embedded Deployment
1. **Generate C Header**: Convert `gesture_model_int8.tflite` to C array
2. **Compile Diagnostic Sketch**: Minimal runtime test on ESP32-WROOM
3. **Confirm Schema Compatibility**: Verify input/output tensor types and dimensions
4. **Check Tensor Allocation**: Measure memory requirements for tensors and intermediates
5. **Verify Operator Support**: Confirm all required operators available in TFLite Micro ESP32
6. **Controlled Input Test**: Feed test input and validate successful inference
7. **Integrate Normalization**: Load saved statistics into embedded preprocessing pipeline
8. **MPU6050 Acquisition**: Implement 50 Hz sensor reading via I2C (GPIO 21/22)
9. **Windowing Assembly**: Buffer 50-sample windows matching training procedure
10. **Real-World Testing**: Evaluate latency, memory consumption, classification stability
11. **Long-Term Deployment**: Assess performance over extended operation periods

### Memory Considerations for ESP32
- **Model Flash**: 43,056 bytes (INT8 TFLite)
- **Model Header**: ~258 KB textual size (hex array representation)
- **Tensor Arena**: Must accommodate input, intermediate, and output tensors
- **ESP32 RAM**: Typical allocation 64KB-256KB for TensorFlow Micro applications
- **Note**: Actual fit must be confirmed via compilation and memory reporting

## Reproducibility Instructions

### To Reproduce Training
```bash
# From LABCA directory:
python3 train_edgeai.py
```
Requirements: TensorFlow 2.22.0-rc0, NumPy, Pandas, scikit-learn, matplotlib

### To Reproduce Quantization
```bash
# From LABCA directory:
python3 quantize_edgeai.py
```
Prerequisites: Training must be completed first (model artifacts present)

### Key Reproducibility Controls
- Fixed random seed: 42 (applied to Python random, NumPy, TensorFlow)
- Deterministic window-level splitting (60/20/20 stratified by class)
- Training-only normalization statistics fitting
- Early stopping with best weight restoration
- Ordered layer architecture as specified

## Known Limitations and Risks

### Data Limitations
1. Single-session data collection (temporal correlations possible)
2. Single participant gesturing style
3. Limited environmental variability (indoor, static background)
4. Small sample size (65 windows total)
5. Potential class imbalance after cleaning (wrist_rotate: 15 vs others: 10)

### Technical Limitations
1. Perfect test score may indicate overfitting to small dataset
2. Quantization verified only on 13-window test set
3. ESP32 operator support and memory footprint unverified
4. Real-world sensor noise and drift not characterized
5. No testing across temperature ranges or voltage variations

### Mitigation Recommendations
1. Collect multi-session, multi-participant dataset
2. Implement leave-one-subject-out cross-validation
3. Test quantization robustness with larger calibration set
4. Profile actual ESP32 memory usage before final integration
5. Consider data augmentation techniques for limited datasets
6. Plan for periodic recalibration in deployed systems

## End-to-End Pipeline Diagram

```mermaid
flowchart TD
    A[Raw Sensor Data] --> B[Data Cleaning]
    B --> C[Windowing & Labeling]
    C --> D[Train/Val/Test Split]
    D --> E[Normalization (Train-only fit)]
    E --> F[Model Training]
    F --> G[Keras Model Evaluation]
    G --> H[Float32 TFLite Conversion]
    H --> I[INT8 Quantization]
    I --> J[C Header Generation]
    J --> K[ESP32 Firmware Integration]
    K --> L[MPU6050 Acquisition @ 50Hz]
    L --> M[Windowing (50 samples)]
    M --> N[Normalization (Saved Stats)]
    N --> O[INT8 Quantization (Firmware)]
    O --> P[TFLite Micro Inference]
    P --> Q[Gesture Classification]
    Q --> R[Action/Output]
```

## Concise Summary

The pipeline achieves perfect accuracy on a small, cleaned pilot dataset (65 windows → 13-window test set) using a lightweight 1D CNN architecture. The model successfully converts to INT8 TFLite with zero measured accuracy loss on the test set, reducing size from 109KB to 43KB. However, the extremely limited test set size prevents conclusions about real-world performance. 

**Verdict**: The pipeline is technically sound and reproducible for proof-of-concept validation, but requires significantly more diverse data collection before performance claims can be made for deployment. The immediate next step is embedded inference validation on the ESP32-WROOM hardware.

---

*Documentation generated from actual repository inspection on 2026-10-09. All claims verified against existing source code, model artifacts, and reports. No results inferred or fabricated.*