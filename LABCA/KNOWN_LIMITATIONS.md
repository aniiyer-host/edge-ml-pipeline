# Edge AI Gesture Recognition Project - Known Limitations

## Hardware Limitations

### Sensor Specifications
- **MPU6050 Accuracy**: The MPU6050 has inherent noise and bias characteristics typical of MEMS sensors; precision limited to approximately ±0.1g for accelerometer and ±5°/s for gyroscope
- **Temperature Drift**: Sensor readings may drift with temperature changes; no active temperature compensation implemented in current firmware
- **Axis Misalignment**: Small manufacturing variances in axis alignment not corrected; assumes perfect orthogonal axes
- **Dynamic Range**: ±1000°/s gyroscope range selected to avoid saturation during wrist rotation, but extremely fast rotations (>1000°/s) will still clip

### ESP32 Constraints
- **Processing Power**: Limited to approximately 240 MHz dual-core processor; complex ML operations may approach computational limits
- **Memory Limitations**: 520KB SRAM total; must carefully manage memory for model, buffers, and OS
- **Flash Constraints**: 4MB flash available but model+firmware+filesystem must fit within usable space
- **Power Consumption**: Continuous WiFi operation for UDP transmission draws significant current (~80-120mA); battery life limited for wearable use

### Physical Implementation
- **Sensor Placement**: Assumes consistent mounting position on wrist/forearm; performance degrades with changes in sensor orientation or location
- **Cable Strain Relief**: Prototyping uses jumper wires on breadboard; not suitable for long-term wearable use without proper strain relief
- **Electrical Noise**: Breadboard prototyping susceptible to electrical interference; final implementation would require PCB with proper grounding
- **Water Resistance**: Current implementation not water-resistant or sweat-proof; limits use in active scenarios

## Machine Learning Limitations

### Data-Related Constraints
- **Dataset Size**: Current pilot dataset limited in size (~60 windows total); may not capture full variability of gesture execution
- **Subject Dependence**: Primarily single-user data; model may not generalize well to different users with varying biomechanics
- **Gesture Consistency**: Natural variation in gesture execution (speed, amplitude, trajectory) may exceed training data coverage
- **Temporal Assumptions**: Fixed 1-second window assumes all gestures complete within this duration; shorter/longer gestures may be misclassified

### Model Limitations
- **Architecture Simplicity**: While parameter-efficient, the 1D-CNN may not capture extremely complex temporal dependencies compared to larger architectures
- **Quantization Impact**: INT8 quantization introduces small accuracy degradation; not characterized for all gesture classes equally
- **Class Separability**: Some gesture pairs may have overlapping temporal signatures (e.g., certain wave variations)
- **Threshold Sensitivity**: Classification dependent on confidence threshold; improper tuning may increase false positives or negatives
- **No Rejection Mechanism**: Model always predicts one of the six classes; no "unknown gesture" or "noise" classification capability

### Training Limitations
- **Single Modality**: Uses only IMU data; no fusion with other potential sensors (camera, EMG, etc.) for improved accuracy
- **Offline Training**: Training performed offline on PC; no on-device learning or adaptation capabilities
- **Fixed Preprocessing**: StandardScaler parameters fixed after training; no adaptation to sensor drift or changes
- **Evaluation Metrics**: Primary reliance on accuracy; may mask class-specific performance issues in imbalanced scenarios

## System Limitations

### Latency Constraints
- **End-to-End Delay**: Approximately 100-200ms from gesture completion to action execution; may feel sluggish for highly interactive applications
- **Variable Latency**: Jitter in wireless transmission and OS scheduling may cause inconsistent response times
- **Windowing Delay**: Inherent 1-second minimum latency due to window-based processing; cannot detect gestures until full window collected

### Reliability Factors
- **Wireless Dependence**: UDP WiFi connection required; performance degrades with packet loss or interference
- **ESP32 Stability**: Extended operation may encounter memory leaks or crashes requiring reboot
- **PC Dependence**: Current implementation relies on PC for inference; not truly standalone edge device
- **Environmental Sensitivity**: Performance may vary with electromagnetic interference, temperature fluctuations, or physical obstructions

### Usability Constraints
- **Learning Curve**: Users must learn specific gesture repertoire; no natural gesture discovery
- **Fatigue Potential**: Repeated gestures may cause muscle fatigue during extended use
- **Environmental Constraints**: Requires line-of-sight to WiFi access point; limited range in obstructed environments
- **Social Acceptability**: Visible gestures may not be appropriate in all social or professional settings

## Scope Limitations

### Functional Scope
- **Gesture Vocabulary Limited to Six Classes**: Does not support complex gestures, sequences, or sustained poses
- **No Context Awareness**: System does not adapt behavior based on application state or user context
- **No Multi-User Support**: Designed for single-user operation; no gesture isolation between nearby users
- **Static Mapping**: Gesture-to-action mappings fixed at runtime; no dynamic reconfiguration without recompilation

### Performance Scope
- **Accuracy Ceiling**: Limited by sensor noise, inter-subject variability, and gesture similarity; unlikely to reach >98% accuracy in uncontrolled environments
- **Throughput Limited**: Maximum gesture recognition rate limited by window size (~1 gesture/second minimum spacing)
- **No Continuous Control**: Recognizes discrete gestures only; no continuous control or trajectory tracking
- **Environmental Robustness**: Not tested in high-interference environments (e.g., near motors, strong magnetic fields)

### Technical Scope
- **Protocol Specificity**: Uses UART→UDP→MQTT stack; not easily adaptable to other communication paradigms
- **Platform Specificity**: Firmware targeted at ESP32; porting to other microcontrollers would require significant changes
- **Software Stack Dependency**: Relies on specific versions of TensorFlow Lite, ESP-IDF, and Python libraries
- **Limited Diagnostics**: Minimal error reporting and diagnostic capabilities in field-deployed state

## Mitigation Strategies (Future Work)

### Hardware Improvements
1. **Custom PCB**: Replace breadboard with properly designed PCB for noise reduction and reliability
2. **Enclosed Housing**: Develop ergonomic, sweat-resistant enclosure for wearable deployment
3. **Battery Optimization**: Implement deep sleep modes and efficient polling to extend battery life
4. **Sensor Calibration**: Add on-demand calibration procedures to compensate for bias and drift
5. **Alternative Sensors**: Evaluate higher-performance IMUs or multi-modal sensing approaches

### ML Enhancements
1. **Larger Dataset**: Collect multi-session, multi-user dataset to improve generalization
2. **Data Augmentation**: Implement synthetic data generation to increase effective dataset size
3. **Model Architectures**: Explore hybridCNN-RNN or attention mechanisms for improved temporal modeling
4. **Online Learning**: Implement lightweight adaptation techniques for user-specific fine-tuning
5. **Uncertainty Estimation**: Add confidence metrics and rejection thresholds for low-quality predictions

### System Improvements
1. **On-Device Inference**: Migrate TFLite model to ESP32 using TensorFlow Lite Micro for true edge deployment
2. **Protocol Optimization**: Implement more reliable wireless protocol with acknowledgments and retransmission
3. **Edge Cases Handling**: Add timeout mechanisms and watchdog timers for improved reliability
4. **User Interface**: Develop haptic or visual feedback to confirm gesture recognition
5. **Configuration System**: Implement runtime configuration via Bluetooth or WiFi for gesture mapping changes

## Assumptions That Limit Generalizability

1. **UiD Assumption**: Assumes gestures performed by same individual as training data
2. **Environmental Assumption**: Assumes typical indoor EM environment without strong interference
3. **Mounting Assumption**: Assumes consistent sensor-to-limb mounting geometry and tight coupling
4. **Temporal Assumption**: Assumes gesture dynamics primarily exist within 0-2Hz frequency band
5. **Isolation Assumption**: Assumes minimal electromagnetic interference from device electronics
6. **Stability Assumption**: Assumes relatively stable positioning during gesture execution (no walking/running)
7. **Lighting Assumption**: Assumes operation not affected by varying light conditions (relevant for camera-based alternatives)
8. **Temperature Assumption**: Assumes operation within 0-40°C range; extreme temperatures not characterized

These limitations represent known constraints based on current implementation and testing. They delineate the boundary between what the system can reliably do versus what would require further development or represent inherent trade-offs in the chosen approach.