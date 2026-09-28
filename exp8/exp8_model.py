import numpy as np
import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# --------------------------------------------------
# 1. Create training data
# --------------------------------------------------
#
# LDR behavior:
# 1 = NO LASER
# 0 = LASER DETECTED
#
# We add repeated samples so the model has
# enough examples to train on.

X = np.array([
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],

    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
], dtype=np.float32)

# Class labels:
# 0 = NO LASER
# 1 = LASER DETECTED

y = np.array([
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0
], dtype=np.int32)

# --------------------------------------------------
# 2. Create tiny neural network
# --------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------
# 3. Train
# --------------------------------------------------

model.fit(
    X,
    y,
    epochs=50,
    verbose=0
)

loss, accuracy = model.evaluate(X, y, verbose=0)

print("\nModel accuracy:", accuracy)
print("Model loss:", loss)

# --------------------------------------------------
# 4. Convert to INT8 TensorFlow Lite
# --------------------------------------------------

def representative_dataset():
    for value in [0.0, 1.0]:
        yield [np.array([[value]], dtype=np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# --------------------------------------------------
# 5. Save model
# --------------------------------------------------

with open("laser_classifier_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("\nCreated:")
print("laser_classifier_int8.tflite")

print("Model size:", len(tflite_model), "bytes")
print("Model size:", len(tflite_model) / 1024, "KB")

# --------------------------------------------------
# 6. Inspect INT8 model
# --------------------------------------------------

import tempfile
import os

temp_model = "temp_laser_model.tflite"

with open(temp_model, "wb") as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(
    model_path=temp_model,
    experimental_delegates=[]
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nINPUT DETAILS")
print("------------")
print("Shape:", input_details[0]["shape"])
print("Type:", input_details[0]["dtype"])
print("Quantization:", input_details[0]["quantization"])

print("\nOUTPUT DETAILS")
print("-------------")
print("Shape:", output_details[0]["shape"])
print("Type:", output_details[0]["dtype"])
print("Quantization:", output_details[0]["quantization"])

os.remove(temp_model)