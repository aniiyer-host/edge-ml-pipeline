import tensorflow as tf
import numpy as np

# Load trained TinyNN model
model = tf.keras.models.load_model("TinyNN_Iris.keras")

# Load training data for representative dataset
X_train = np.load("X_train.npy").astype(np.float32)

# Representative dataset for INT8 calibration
def representative_dataset():
    for sample in X_train[:100]:
        yield [sample.reshape(1, -1)]

# Create TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable full INT8 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# Force INT8 input/output
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert
tflite_model = converter.convert()

# Save model
with open("TinyNN_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("TinyNN_int8.tflite created successfully!")
print(f"Model size: {len(tflite_model) / 1024:.2f} KB")