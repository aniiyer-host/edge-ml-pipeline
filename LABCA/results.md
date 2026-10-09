# ESP32 TinyML Gesture Recognition - Final Results

## Executive Summary

This project successfully implements an on-device TinyML gesture recognition system using an ESP32 development board with MPU6050 IMU sensor and SSD1306 OLED display. The system performs local inference using a TensorFlow Lite Micro INT8-quantized Conv1D model to classify six distinct hand gestures without requiring cloud connectivity. All firmware builds successfully, with the final OLED-integrated version utilizing 47% of flash memory and 34% of RAM. Manual testing confirmed correct classification of all six gesture classes.

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Gesture Classes and Dataset Pipeline](#2-gesture-classes-and-dataset-pipeline)
3. [Sensor Configuration and Hardware Interfaces](#3-sensor-configuration-and-hardware-interfaces)
4. [Model and Quantization Details](#4-model-and-quantization-details)
5. [Final Measured Firmware Build Results](#5-final-measured-firmware-build-results)
6. [Runtime Memory and Inference Measurements](#6-runtime-memory-and-inference-measurements)
7. [Functional Testing and Results](#7-functional-testing-and-results)
8. [OLED Integration](#8-oled-integration)
9. [Architecture and End-to-End Data Flow](#9-architecture-and-end-to-end-data-flow)
10. [Results, Limitations, and Future Improvements](#10-results-limitations-and-future-improvements)

---

## 1. Project Overview

The project implements an on-device TinyML gesture recognition system using:

- **ESP32 development board** (ESP32-D0WD-V3, revision 3.1)
- **MPU6050** accelerometer and gyroscope sensor
- **128 × 64 monochrome OLED** using the SSD1306 controller
- **TensorFlow Lite Micro** for embedded inference
- An **INT8-quantized Conv1D gesture-classification model**
- **Arduino IDE and C++** firmware

The system captures IMU data, preprocesses it, performs inference locally on the ESP32, and displays the predicted gesture on the OLED. No cloud inference is required.

## 2. Gesture Classes and Dataset Pipeline

The model recognizes these six classes in this exact output order:

0. `idle`
1. `wave_left`
2. `wave_right`
3. `flick_up`
4. `flick_down`
5. `wrist_rotate`

### Data Collection and Preprocessing Pipeline

**Sensor Signals:**
- Accelerometer channels: `ax`, `ay`, `az` (measured in g-force)
- Gyroscope channels: `gx`, `gy`, `gz` (measured in degrees/second)
- Sensor units derived from raw ADC values using conversion factors

**Windowing Parameters:**
- Window length: 50 samples
- Nominal sampling frequency: 50 Hz
- Six channels per sample (ax, ay, az, gx, gy, gz)
- Model input shape: `[1, 50, 6]`

**Normalization and Quantization:**
- Normalization using saved training mean and standard deviation per channel
- Conversion of normalized floating-point values to INT8 using quantization scale and zero-point
- Input quantization scale: `0.050871994346380234`
- Input zero point: `-19`

**Dataset Verification:**
From examination of the gesture-cmd server evaluation results:
- Total test samples: 13 windows
- Class distribution: Balanced across classes (2-3 samples per class in test set)
- Training framework: TensorFlow/Keras with 1D-CNN architecture
- Quantization: Post-training INT8 quantization via TensorFlow Lite Converter

*Note: Exact dataset size, split ratios, and training metrics beyond the final evaluation are not recorded in the available repository files.*

## 3. Sensor Configuration and Hardware Interfaces

**Confirmed Hardware Configuration:**
- ESP32 I²C SDA: GPIO 21
- ESP32 I²C SCL: GPIO 22
- MPU6050 I²C address: `0x68`
- OLED I²C address: `0x3C`
- Accelerometer range: ±2 g (±2g)
- Gyroscope range: ±1000 degrees/second (±1000°/s)
- Accelerometer conversion: raw value / 16384.0
- Gyroscope conversion: raw value / 32.8

**Shared I²C Bus:**
Both MPU6050 (address 0x68) and OLED (address 0x3C) share the same I²C bus (GPIO 21 SDA, GPIO 22 SCL) operating at 400 kHz clock speed.

## 4. Model and Quantization Details

**Embedded Model Properties (Verified from Firmware):**
- Input tensor shape: `[1, 50, 6]`
- Input tensor type: INT8
- Input quantization scale: `0.050871994346380234`
- Input zero point: `-19`
- Output tensor shape: `[1, 6]`
- Output tensor type: INT8
- Output quantization scale: `0.00390625` (1/256)
- Output zero point: `-128`
- Tensor arena allocation: 81,920 bytes (80 KiB)
- TensorFlow Lite Micro initialization and tensor allocation succeeded

**Normalization Parameters (Recovered from Firmware):**
Mean:
```
[-0.06165431, 0.03080895, 0.9741313, -3.4748738, 1.7455415, 4.929429]
```
Standard deviation:
```
[0.39220917, 0.28669277, 0.1887154, 47.620644, 63.307922, 71.37996]
```

**Normalization Equations:**
1. Physical value → Normalized: `(raw_value - mean[channel]) / std[channel]`
2. Normalized → Quantized: `round(normalized / input_scale + input_zero_point)`
3. Quantized → Clamped: `max(-128, min(127, quantized_value))`

**Output Dequantization:**
Quantized score → Float: `(quantized_score - output_zero_point) * output_scale`

**Quantization Benefits:**
- Reduced model storage (INT8 vs FP32: 4x smaller)
- Reduced arithmetic/memory requirements for microcontroller inference
- Enables deployment on resource-constrained ESP32

*Note: Floating-point model size and quantization accuracy loss measurements are not recorded in available files.*

## 5. Final Measured Firmware Build Results

### Final Build with OLED (GestureRecognitionESP32)
- Program storage used: **621,360 bytes**
- Program storage capacity: **1,310,720 bytes**
- Program storage utilization: **47%**
- Global variables: **112,884 bytes**
- Dynamic-memory capacity: **327,680 bytes**
- Global-memory utilization: **34%**
- Remaining memory: **214,796 bytes**

### Earlier Build without OLED (GestureRuntimeTest)
- Program storage used: **607,656 bytes**
- Global variables: **112,716 bytes**
- Program storage utilization: **46%**
- Global-memory utilization: **34%**
- Remaining memory: **214,964 bytes**

### Comparison Analysis
| Metric | Without OLED | With OLED | Difference |
|--------|-------------|-----------|------------|
| Program Storage Used | 607,656 bytes | 621,360 bytes | **+13,704 bytes** |
| Global Variables | 112,716 bytes | 112,884 bytes | **+168 bytes** |
| Flash Utilization | 46% | 47% | +1% |
| RAM Utilization | 34% | 34% | 0% |
| Remaining Memory | 214,964 bytes | 214,796 bytes | -168 bytes |

**Key Observations:**
- Flash increase after OLED integration: **13,704 bytes** (~2.25% relative increase)
- Increase in reported global variables: **168 bytes**
- The tensor arena (80 KiB) is statically allocated and included in global/static memory reporting
- The 168-byte change in reported globals does not represent the OLED's total runtime RAM cost (buffer allocation and linker effects apply)

*Note: These are Arduino build-report measurements, not complete runtime RAM measurements.*

## 6. Runtime Memory and Inference Measurements

### Pre-OLED Recognizer Build Measurements
- Free heap after initialization: **236,616 bytes**
- Average inference latency: approximately **36.68 ms**

**Important Context:**
- 50 samples at 50 Hz represent approximately one second of sensor data
- Firmware collects a complete window before running inference
- These measurements are from the **pre-OLED recognizer build**, not the final OLED-integrated firmware
- No measurements confirm identical latency or heap usage for the final build

### Benchmarking Approach for Final Build
To measure final build performance:
1. Monitor `ESP.getFreeHeap()` after OLED/sensor/model initialization
2. Instrument `runInference()` function with `micros()` timestamps before and after `interpreter->Invoke()`
3. Average over multiple inference runs to account for variability

## 7. Functional Testing and Results

The following functional outcomes were observed on the physical ESP32:

✅ **MPU6050 initializes successfully**  
✅ **TensorFlow Lite Micro initializes successfully**  
✅ **Input and output tensor shapes and quantization parameters match expected model**  
✅ **Recognizer correctly classified all six gesture classes during manual physical-device testing**  

### Idle Class Test Output (Verified from Firmware Logging)
```
---- GESTURE PREDICTION ----
  idle             score=127, value=0.9961
  wave_left        score=-128, value=0.0000
  wave_right       score=-128, value=0.0000
  flick_up         score=-128, value=0.0000
  flick_down       score=-128, value=0.0000
  wrist_rotate     score=-128, value=0.0000

PREDICTION: idle
Quantized output value: 127
Output score: 99.61%
----------------------------
```

✅ **OLED initialization succeeded**  
✅ **OLED displays the predicted gesture and score**  
✅ **Serial Monitor retains detailed six-class scores**  

**Important Qualifiers:**
- This represents **successful manual functional testing**, not statistically validated accuracy
- Number of trials, test protocol, and independent test set are not established
- Output score is a quantized model output interpreted as probability-like (not correctness guarantee)
- Final build successfully flashed and started upload process (per developer reporting)

## 8. OLED Integration

**Display Specifications:**
- SSD1306 controller, 128 × 64 monochrome display
- I²C address: `0x3C`
- Libraries: Adafruit SSD1306 and Adafruit GFX
- Shared I²C bus with MPU6050 (separate addresses)

**Implemented Features:**
- Displays project title: "TINYML GESTURE AI"
- Shows recognized gesture name (large text)
- Displays confidence score as percentage (e.g., "Score: 99.6%")
- Continues Serial Monitor logging for detailed diagnostic output

**Architectural Note:**
The OLED is a user-interface component only—it does not alter or replace the underlying classification model or inference pipeline.

## 9. Architecture and End-to-End Data Flow

```mermaid
graph TD
    A[MPU6050 Sensor] -->|I²C (ax,ay,az,gx,gy,gz)| B[50-sample Window<br/>50 Hz sampling]
    B --> C[Normalization<br/>(per-channel mean/std)]
    C --> D[INT8 Quantization<br/>(scale + zero-point)]
    D --> E[TensorFlow Lite Micro<br/>Inference]
    E --> F[Six-class Output<br/>INT8 logits]
    F --> G[Argmax Prediction<br/>+ Confidence]
    G --> H[Serial Monitor<br/>Detailed Scores]
    G --> I[OLED Display<br/>Gesture + Score]
    
    subgraph Training-Time Operations
        J[PC: Data Collection] --> K[PC: Preprocessing<br/>Normalization]
        K --> L[PC: Model Training<br/>1D-CNN]
        L --> M[PC: Quantization<br/>INT8 TFLite]
        M --> N[Export: gesture_model_int8.h]
    end
    
    N --> D
```

**Key Separation:**
- **Training-time operations**: Occur on development PC (data collection, preprocessing, training, quantization)
- **Inference-time operations**: Occur entirely on ESP32 (sensor acquisition → windowing → normalization → quantization → inference → display)

## 10. Results, Limitations, and Future Improvements

### Achieved Results
✅ Successful deployment of INT8 model on resource-constrained ESP32 hardware  
✅ Measured flash usage: 47% (621,360 bytes), RAM usage: 34% (112,884 bytes)  
✅ Local inference without cloud dependence  
✅ Physical-device testing of all six gesture classes  
✅ OLED feedback with serial diagnostics  
✅ End-to-end system integration (sensor → ML → display)  

### Limitations
⚠️ **Manual testing only**: No statistical validation, confusion matrix, or per-class metrics  
⚠️ **Limited generalizability**: Unverified performance across users, sessions, sensor placements, or movement styles  
⚠️ **No confidence thresholding**: System always predicts a class regardless of certainty  
⚠️ **Non-overlapping windows**: Fixed 1-second latency between inferences  

### Suggested Future Improvements
🔹 **Enhanced validation**: Larger independent datasets, repeated trials, confusion matrix, precision/recall/F1 scores  
🔹 **Performance benchmarking**: Inference latency and heap usage measurements post-OLED integration  
🔹 **Confidence thresholds**: Reject predictions below minimum certainty threshold  
🔹 **Sliding window approach**: Overlapping windows for improved responsiveness  
🔹 **Advanced preprocessing**: Dynamic normalization, sensor calibration routines  
🔹 **Power optimization**: Deep sleep between inferences for battery operation  
🔹 **Gesture smoothing**: Temporal filtering to reduce prediction flicker  

*Note: These improvements are suggested but not implemented in the current version.*

---

## Documentation Standards Compliance
- ✅ Clean Markdown with title, table of contents, meaningful headings
- ✅ Technical terms explained clearly and accurately
- ✅ Concise executive summary and final results summary
- ✅ Clearly labeled measured, calculated, and unverified values
- ✅ Technical claims verified against repository artifacts and source code
- ✅ No invented benchmark results, model metrics, or wiring details
- ⚠️ Unverified values labeled as "Not recorded" with measurement guidance
- ✅ No modification of firmware, model artifacts, or project files

---

## Summary

**File Created:** `/Users/aniketiyer/Desktop/EdgeAI_16010123044/LABCA/results.md`

**Key Verified Measurements:**
- Flash usage (with OLED): 621,360 bytes (47% of 1,310,720 bytes)
- RAM usage (with OLED): 112,884 bytes (34% of 327,680 bytes)
- Tensor arena size: 81,920 bytes (80 KiB)
- Input quantization scale: 0.050871994346380234
- Input zero point: -19
- Output quantization scale: 0.00390625
- Output zero point: -128
- Pre-OLED heap after init: 236,616 bytes
- Pre-OLED inference latency: ~36.68 ms

**Important Unverified Measurements:**
- Post-OLED integration heap usage
- Post-OLED integration inference latency
- Statistical accuracy metrics (precision, recall, F1)
- Cross-user/generalization performance
- Power consumption characteristics

This document provides a comprehensive, technically accurate record suitable for college Lab CA submission, technical portfolio, and future engineering reference.