# Edge AI Gesture Recognition Project - Model Architecture Explanation

## Overview

The selected model for gesture recognition is a lightweight 1D Convolutional Neural Network (1D-CNN) specifically designed for temporal IMU data classification. This architecture was chosen after evaluating alternatives including Decision Trees, Random Forest, SVM, and Gaussian Naive Bayes, with the 1D-CNN proving superior for capturing the temporal dynamics essential to gesture recognition.

## Architectural Rationale

### Why 1D-CNN for Gesture Recognition?

1. **Temporal Nature of Data**: IMU gesture data is inherently temporal - a gesture is not defined by instantaneous sensor values but by how those values change over time. A 1D-CNN excels at learning local temporal patterns.

2. **Local Feature Extraction**: Convolutional kernels slide across the time dimension to detect short-term patterns such as:
   - Acceleration peaks and valleys
   - Rotational onset and cessation
   - Direction changes in motion
   - Cross-axis relationships (e.g., when gyroscope X increases while accelerometer Z decreases)

3. **Parameter Efficiency**: Compared to fully connected networks or RNNs, 1D-CNNs achieve strong temporal modeling with significantly fewer parameters, making them suitable for Edge AI deployment.

4. **Translation Invariance**: Convolutional features are somewhat invariant to exact temporal positioning within the window, which aligns with the goal of recognizing gestures regardless of their precise timing within the 1-second window.

## Detailed Architecture

The model processes input windows of shape `(50, 6)` representing 50 samples of 6 IMU channels (ax, ay, az, gx, gy, gz).

### Layer-by-Layer Breakdown

#### Input Layer
- **Shape**: `(50, 6)`
- **Description**: Raw IMU time series data
- **Processing**: No modification - data flows directly to first convolutional layer

#### Block 1: Initial Feature Extraction
```
Conv1D(32 filters, kernel_size=5, padding='same', activation='relu')
BatchNormalization()
MaxPooling1D(pool_size=2)
Dropout(rate=0.3)
```
- **Conv1D**: 32 kernels of width 5 slide across time dimension
  - Each kernel learns to detect specific 5-sample temporal patterns
  - Padding='same' preserves temporal dimension (50 → 50)
  - Output shape: `(50, 32)`
- **BatchNormalization**: Normalizes activations per feature map
  - Stabilizes and accelerates training
  - Reduces internal covariate shift
- **MaxPooling1D**: Downsamples by factor of 2
  - Takes maximum value in each 2-sample window
  - Provides translational invariance
  - Reduces computational load
  - Output shape: `(25, 32)`
- **Dropout**: Randomly zeroes 30% of activations during training
  - Prevents overfitting
  - Acts as model averaging

#### Block 2: Feature Refinement
```
Conv1D(64 filters, kernel_size=3, padding='same', activation='relu')
BatchNormalization()
MaxPooling1D(pool_size=2)
Dropout(rate=0.3)
```
- **Conv1D**: 64 kernels of width 3
  - Builds upon features from Block 1
  - Learns more complex patterns from combinations of simpler features
  - Output shape: `(25, 64)`
- **BatchNormalization**: Stabilizes layer outputs
- **MaxPooling1D**: Further downsamples by factor of 2
  - Output shape: `(12, 64)`
- **Dropout**: Continues regularization (30% rate)

#### Block 3: High-Level Feature Learning
```
Conv1D(64 filters, kernel_size=3, padding='same', activation='relu')
BatchNormalization()
Dropout(rate=0.3)
```
- **Conv1D**: 64 kernels of width 3
  - Learns sophisticated temporal patterns
  - No pooling to preserve more temporal information for classification
  - Output shape: `(12, 64)`
- **BatchNormalization**: Maintains training stability
- **Dropout**: Final regularization before pooling

#### Temporal Pooling
```
GlobalAveragePooling1D()
```
- **Operation**: Averages each feature map across the entire time dimension
- **Input**: `(12, 64)`
- **Output**: `(64,)` - 128-dimensional vector
- **Advantages**:
  - Dramatically reduces parameters compared to flattening
  - Preserves channel-wise information while removing temporal ordering
  - Makes model invariant to temporal translations of features
  - Reduces final Dense layer input size from 12*64=768 to just 64

#### Classification Head
```
Dense(64, activation='relu')
Dropout(rate=0.3)
Dense(n_classes, activation='softmax')
```
- **Dense(64)**: Learns complex mappings from features to gesture classes
  - 64 units provide sufficient capacity for 6-class classification
  - ReLU activation enables nonlinear decision boundaries
- **Dropout**: Final regularization (30% rate)
- **Dense(n_classes)**: Output layer with softmax activation
  - n_classes = 6 for final gesture set
  - Output: probability distribution over gesture classes
  - Softmax ensures outputs sum to 1.0

## Parameter Count Analysis

Let's calculate the approximate parameter count:

### Block 1
- Conv1D(32, k=5): (5 * 6 * 32) + 32 biases = 960 + 32 = 992
- BatchNorm: 2 * 32 = 64 (gamma and beta parameters)
- Total Block 1: ~1,056 parameters

### Block 2
- Conv1D(64, k=3): (3 * 32 * 64) + 64 biases = 6,144 + 64 = 6,208
- BatchNorm: 2 * 64 = 128
- Total Block 2: ~6,336 parameters

### Block 3
- Conv1D(64, k=3): (3 * 64 * 64) + 64 biases = 12,288 + 64 = 12,352
- BatchNorm: 2 * 64 = 128
- Total Block 3: ~12,480 parameters

### Classification Head
- Dense(64): (12 * 64 * 64) + 64 biases = 49,152 + 64 = 49,216
  - Note: Actually connects to GAP output of 64, not 12*64
  - Correction: (64 * 64) + 64 = 4,096 + 64 = 4,160
- Dense(6): (64 * 6) + 6 biases = 384 + 6 = 390
- Total Head: ~4,550 parameters

### Total Parameters
~1,056 + 6,336 + 12,480 + 4,550 = **24,422 parameters**

This aligns with the stated target of approximately 25K parameters.

## Design Choices Justification

### Kernel Sizes
- **First layer (k=5)**: Captures longer-term patterns (~100ms at 50Hz) needed for gesture onset/termination detection
- **Subsequent layers (k=3)**: Focus on shorter, more precise temporal relationships as abstraction level increases

### Filter Progression (32 → 64 → 64)
- Starts with fewer, broader features
- Increases to capture more complex combinations
- Maintains 64 in final layer to preserve representational power before pooling

### Pooling Strategy
- Uses MaxPooling in intermediate layers for dimensionality reduction while preserving salient features
- Employs GlobalAveragePooling1D at the end to drastically reduce parameters before Dense layers
- This combination is more parameter-efficient than using only MaxPooling or only average pooling

### Dropout Rate (0.3)
- Empirically chosen balance between regularization and model capacity
- High enough to prevent overfitting on limited dataset
- Low enough to preserve learning capacity

### Batch Normalization
- Applied after each Convolutional layer before activation
- Stabilizes distribution of layer inputs
- Allows higher learning rates and reduces sensitivity to initialization

### Global Average Pooling
- Critical innovation that enables small model size
- Instead of flattening (which would create 12*64=768 inputs to first Dense layer),
- GAP reduces to just 64 inputs, saving ~700 parameters in the first Dense layer alone
- This single architectural choice is largely responsible for achieving the <25K parameter target

## Activation Functions

### ReLU (Rectified Linear Unit)
- Used in all hidden layers (Conv1D and Dense)
- **Formula**: f(x) = max(0, x)
- **Advantages**:
  - Computationally efficient (simple thresholding)
  - Mitigates vanishing gradient problem
  - Induces sparsity in activations
  - Empirically effective for deep architectures

### Softmax
- Used only in final output layer
- **Formula**: softmax(z_i) = e^(z_i) / Σ_j e^(z_j)
- **Purpose**: Converts logits to probability distribution over classes
- **Properties**: Outputs sum to 1.0, differentiable, preserves rank order of inputs

## Why Not Alternative Architectures?

### Dense Network (MLP)
- Would require flattening input to 300-dimensional vector
- First hidden layer of 100 units would have 300*100 + 100 = 30,100 parameters just for input
- No temporal locality exploitation - treats time steps as independent features
- Significantly higher parameter count for equivalent performance

### RNN/LSTM
- Sequential processing creates inference latency bottlenecks
- Significantly more complex than Conv1D (gates, cell states)
- Parameter overhead from recurrent connections
- Difficult to parallelize efficiently on microcontrollers
- Overkill for relatively short sequences (50 timesteps)

### 2D CNN
- Inappropriate for 1D temporal data
- Would require artificially structuring data as "image"
- No spatial locality in IMU channels to exploit
- Unnecessarily complex kernel designs

### Tree-Based Methods (DT, RF, GNB)
- Cannot automatically learn temporal features
- Require manual feature engineering (mean, std, FFT, etc.)
- Poor at capturing temporal dependencies and local patterns
- Generally higher prediction latency than equivalent CNNs
- No natural way to exploit translation invariance

## Quantization Considerations

The architecture was designed with eventual INT8 quantization in mind:

### Quantization-Friendly Properties
- **ReLU Activation**: Outputs naturally non-negative, simplifying quantization
- **Batch Normalization**: Can be folded into preceding Convolutional layers
- **Global Average Pooling**: Involves only averaging - stable under quantization
- **Moderate Filter Counts**: Avoids extremely large or small values that quantization struggles with
- **Reasonable Depth**: 3 Conv blocks provides sufficient feature hierarchy without excessive quantization error accumulation

### Quantization Process
1. **Training**: Float32 model achieves baseline accuracy
2. **Calibration**: Representative dataset (~200 samples) determines activation ranges
3. **Conversion**: 
   - Weights converted from float32 to int8 using per-tensor scaling
   - Activations quantized dynamically during inference using layer-specific scale/zero_point
   - Arithmetic implemented using integer operations with requantization steps
4. **Validation**: Quantized model accuracy compared to float32 baseline

## Deployment Implications

### Computational Complexity
- **Operations**: Dominated by Convolutional layers
- **Memory**: 
  - Weights: ~24K parameters × 1 byte (int8) = ~24KB
  - Activations: Intermediate feature maps require additional RAM
  - Total RAM target: <50KB for inference on ESP32
- **Cycle Count**: Estimated <10 million MAC operations for full inference

### Latency Breakdown (Estimated)
- Sensor sampling: 20ms per sample (50Hz)
- Window acquisition: 1000ms (50 samples)
- Preprocessing (normalization): <1ms
- Model inference: <10ms (target)
- Total window-to-decision: ~1010ms
- Effective latency: <110ms from window completion to gesture recognition

### Memory Footprint
- **Flash Storage**:
  - Model weights: ~24KB (int8 TFLite)
  - Model definition: <2KB
  - Label mapping: <1KB
  - **Total flash**: <30KB
- **RAM Usage**:
  - Input window: 50×6×4 bytes = 1.2KB (float32)
  - Intermediate activations: ~5-10KB
  - Output probabilities: 6×1 byte = 6 bytes
  - Stack & overhead: ~2-5KB
  - **Total RAM**: <20KB (leaving ample headroom for other tasks)

## Architectural Innovations for Edge AI

### 1. Parameter Efficiency Through Architectural Choices
The combination of strided convolutions, pooling, and Global Average Pooling achieves strong feature extraction with minimal parameters - a critical constraint for deployment on microcontrollers.

### 2. Bridge Between Raw Signals and Semantic Concepts
Instead of relying on hand-crafted features, the CNN learns hierarchical representations:
- Layer 1: Edges, onsets, offsets in individual channels
- Layer 2: Combinations (e.g., accel peak + gyro zero-crossing)
- Layer 3: Higher-order motion patterns (e.g., rotary oscillations)
- Classification: Mapping of motion patterns to gesture semantics

### 3. Robustness Through Regularization
The combination of BatchNorm and Dropout provides complementary regularization:
- BatchNorm reduces internal covariate shift
- Dropout prevents co-adaptation of features
- Together they significantly improve generalization from limited training data

### 4. Deployment-Aware Design
Every architectural decision considered eventual deployment on ESP32:
- Kernel sizes chosen for computational efficiency
- Filter counts balanced for representational power vs. size
- Avoidance of operations problematic for microcontroller implementation
- Quantization compatibility maintained throughout

## Limitations and Future Work

### Current Limitations
1. **Fixed Window Size**: Assumes gestures fit within 1-second windows; longer/shorter gestures may be truncated or padded with idle
2. **Subject Dependency**: Trained and tested primarily on single user; inter-subject variability not extensively characterized
3. **Environmental Sensitivity**: Assumes consistent sensor mounting and orientation
4. **Gesture Set Limitation**: Six-class system may not cover all desired interactions

### Potential Improvements
1. **Multi-scale Architecture**: Parallel branches with different kernel sizes to capture patterns at multiple temporal resolutions
2. **Attention Mechanisms**: Lightweight attention to weigh important temporal regions more heavily
3. **Adaptive Windowing**: Variable-length windows based on motion onset/offset detection
4. **Subject Adaptation**: Transfer learning or fine-tuning approaches for new users
5. **Architecture Search**: Neural Architecture Search specifically targeting microcontroller constraints

## Conclusion

The 1D-CNN architecture represents an optimal trade-off for Edge AI gesture recognition:
- Sufficiently expressive to capture complex temporal motion patterns
- Parameter-efficient enough for deployment on resource-constrained microcontrollers
- Designed with quantization compatibility from the outset
- Robust to variations in gesture execution through learned temporal features
- Computationally efficient for real-time inference

This architecture successfully bridges the gap between raw IMU sensor data and meaningful gesture recognition while respecting the stringent constraints of Edge AI deployment.