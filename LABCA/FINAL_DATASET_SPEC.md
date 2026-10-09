# Final Dataset Specification

Based on the LABCA/gesture-cmd dataset format and preprocessing pipeline, this document specifies the final dataset structure used for training and evaluating the gesture recognition model.

## Dataset Format Specification

### File Structure
The dataset is stored as a CSV file with the following columns:
```
label,sample_id,timestamp_ms,sample,ax,ay,az,gx,gy,gz
```

### Column Definitions

| Column Name | Data Type | Description | Example Values |
|-------------|-----------|-------------|----------------|
| **label** | String | Gesture class identifier | "idle", "wave_left", "wrist_rotate" |
| **sample_id** | Integer | Unique identifier for each 50-sample window | 1, 2, 3, ... |
| **timestamp_ms** | Float | Millisecond timestamp of sample acquisition | 0.0, 20.0, 40.0, ... |
| **sample** | Integer | Sample index within the window (0-49) | 0, 1, 2, ..., 49 |
| **ax** | Float | Accelerometer X-axis (g-force) | -0.0520, 0.1234 |
| **ay** | Float | Accelerometer Y-axis (g-force) | 0.0161, -0.4567 |
| **az** | Float | Accelerometer Z-axis (g-force) | 1.1150, 0.9876 |
| **gx** | Float | Gyroscope X-axis (°/second) | -0.4878, 2.3456 |
| **gy** | Float | Gyroscope Y-axis (°/second) | 0.3963, -1.2345 |
| **gz** | Float | Gyroscope Z-axis (°/second) | -0.4268, 0.5678 |

### Windowing Strategy
- **Window Size**: Fixed at 50 samples
- **Sampling Rate**: 50 Hz (20 ms/sample)
- **Window Duration**: 1.0 second (50 samples × 20 ms/sample)
- **Temporal Alignment**: Gestures may occur at any position within the window (early, middle, late) with variable idle periods before/after

### Data Integrity Requirements
1. **Exact Sample Count**: Every window must contain exactly 50 rows (samples 0-49)
2. **Label Consistency**: All rows within a window must have identical label values
3. **Sequential Sampling**: sample column must contain values 0 through 49 in order
4. **Chronological Ordering**: timestamp_ms values must increase monotonically
5. **Numeric Validity**: All sensor values must be valid floating-point numbers
6. **No Missing Data**: No null/NaN/Inf values permitted in any column
7. **Window Completeness**: No partial windows permitted (<50 samples)
8. **Boundary Markers**: Windows delineated by "# BEGIN_WINDOW" and "# END_WINDOW" comments (for human readability, not parsed by ML pipeline)

### Example Window Format
```
# BEGIN_WINDOW
# gesture = wave_left
# window = 1
# samples = 50
# sample_rate_hz = 50
# ----------------------------------------
label,sample_id,timestamp_ms,sample,ax,ay,az,gx,gy,gz
wave_left,1,0,0,-0.0210,0.0034,0.9876,-0.1234,0.5678,-0.0456
wave_left,1,1,20,-0.0198,0.0041,0.9891,-0.1102,0.5891,-0.0332
... (48 more rows) ...
wave_left,1,49,980,0.0345,-0.0012,1.0234,0.8765,-0.2345,0.1122
# END_WINDOW
```

## Gesture Class Definitions

The final dataset contains six mutually exclusive classes:

| Class Name | Description | Distinguishing Characteristics |
|------------|-------------|--------------------------------|
| **idle** | Resting hand position | Minimal acceleration and rotation; gravity vector primarily in Z-axis |
| **wave_left** | Horizontal hand waveLeft-to-right | Periodic Y-axis acceleration with corresponding X-axis gyroscope oscillation |
| **wave_right** | Horizontal hand waveRight-to-left | Periodic Y-axis acceleration with corresponding X-axis gyroscope oscillation (opposite phase) |
| **flick_up** | Vertical upward flick | Sharp positive Z-acceleration pulse followed by decay |
| **flick_down** | Vertical downward flick | Sharp negative Z-acceleration pulse followed by decay |
| **wrist_rotate** | Rotational wrist motion | Sustained gyroscope activity in X and/or Y axes with minimal linear acceleration |

## Data Collection Protocol

### Collection Procedure
1. **Device Setup**: ESP32 equipped with MPU6050 running data collection firmware
2. **Connection**: Serial connection to PC running gesture_logger.py
3. **Labeling Protocol**: 
   - Send 's' via serial to start recording window
   - Perform gesture naturally within the 1-second window
   - Send 'x' to discard window if gesture was mistimed or corrupted
   - Automatic window closure after 50 samples
4. **Environment**: Indoor setting with minimal electromagnetic interference
5. **Sensor Mounting**: Consistent placement on wrist or forearm
6. **Variation Encouraged**: 
   - Different speeds and amplitudes
   - Variable timing within window
   - Different starting/ending positions
   - Natural biomechanical variation

### Quality Control Measures
- **Visual Verification**: Operator confirms correct gesture execution
- **Immediate Feedback**: Real-time serial output shows incoming data
- **Corruption Detection**: Obvious sensor faults visible in serial stream
- **Window Integrity**: Fixed sample count prevents timing ambiguities
- **Rejection Protocol**: Entire window rejected if any sample appears corrupted (no sample-level editing)

## Preprocessing Pipeline Transforms

The raw dataset undergoes the following transformations before model training:

1. **Window Validation**: Ensures 50 samples/window, label consistency, sequential samples
2. **Label Encoding**: String labels mapped to integers (0-5) via LabelEncoder
3. **Train/Val/Test Split**: Stratified 70/15/15 split maintaining class distribution
4. **Feature Standardization**: 
   - StandardScaler fitted ONLY on training data
   - Each feature channel (ax,ay,az,gx,gy,gz) normalized independently
   - Formula: x_normalized = (x - μ_train) / σ_train
   - Same transformation applied to validation and test sets
5. **Tensor Formation**: Data reshaped to (N_windows, 50, 6) for CNN input

## Features Used for Modeling

The model consumes ONLY these six features in this order:
1. ax - Accelerometer X-axis (g)
2. ay - Accelerometer Y-axis (g)
3. az - Accelerometer Z-axis (g)
4. gx - Gyroscope X-axis (°/s)
5. gy - Gyroscope Y-axis (°/s)
6. gz - Gyroscope Z-axis (°/s)

### Features Specifically EXCLUDED from Modeling
- label - Used for supervision only, not model input
- sample_id - Window identifier, not temporal feature
- timestamp_ms - Absolute timing, not used (relative timing within window matters)
- sample - Window-internal index, not used (sequential nature implicit in windowing)

## Storage and Versioning

### File Organization
- **Raw Data**: `dataset.txt` (human-readable with comments)
- **Version Control**: `dataset.csv.dvc` (DVC tracking file pointing to AWS S3)
- **Processed Arrays**: 
  - `X_train.npy`, `X_val.npy`, `X_test.npy` - Normalized input tensors
  - `y_train.npy`, `y_val.npy`, `y_test.npy` - Integer label arrays
- **Metadata**:
  - `label_map.json` - Bidirectional label ↔ index mapping
  - `scaler.pkl` - Fitted StandardScaler for identical preprocessing

### Example DVC File Contents
```
outs:
- md5: f0bb3116c23c5a3ecb8148b7796e4443
  size: 3672855
  hash: md5
  path: gesture_dataset.csv
```

## Usage Guidelines

### For Training
1. Load raw CSV with pandas
2. Validate window integrity (50 samples/window, label consistency)
3. Extract features and labels
4. Apply train/val/test split (stratified)
5. Fit StandardScaler on training data ONLY
6. Transform all sets with fitted scaler
7. Convert labels to integers using LabelEncoder
8. Reshape X to (N_windows, 50, 6)
9. Proceed to model training

### For Inference (Matching Training Preprocessing)
1. Collect 50-sample window from sensor
2. Apply SAME standardization as training:
   - x_normalized = (x - μ_train) / σ_train
   - Using μ_train and σ_train from scaler.pkl
3. Ensure feature order: [ax, ay, az, gx, gy, gz]
4. Reshape to (1, 50, 6) for batch size of 1
5. Feed to model for prediction
6. Apply softmax to logits for class probabilities
7. Select argmax class (if above confidence threshold)

## Limitations and Assumptions

### Known Limitations
1. **Single-Subject Bias**: Initial dataset may not capture inter-subject variability
2. **Environmental Specificity**: Collected in specific indoor EM environment
3. **Mounting Variability**: Assumes consistent sensor-to-limb geometry
4. **Gesture Duration**: Assumes gestures complete within ~1 second window
5. **Sensor Noise**: Subject to MEMS sensor limitations (±0.1g accel, ±5°/s gyro noise)

### Assumptions
1. **Stationary Statistics**: Feature means and standard deviations stable across collection sessions
2. **Gesture Temporal Boundaries**: Most gesture dynamics occur within 1-second window
3. **Class Separability**: Six classes provide sufficient discriminability in IMU space
4. **User Consistency**: Biomechanics of gesture execution similar across users
5. **Environmental Consistency**: Similar indoor environments for deployment and training

This specification ensures reproducible, leakage-free dataset handling while preserving the raw data integrity essential for scientifically valid machine learning research.