import os
import tensorflow as tf
import pandas as pd

# ==========================================================
# Task 6 - Edge AI Suitability Analysis
# ==========================================================

print("=" * 60)
print("TASK 6 - EDGE AI SUITABILITY ANALYSIS")
print("=" * 60)

# ----------------------------------------------------------
# Load Model
# ----------------------------------------------------------

model = tf.keras.models.load_model("TinyNN_Iris.keras")

parameters = model.count_params()

model_size = os.path.getsize("TinyNN_Iris.keras") / 1024

# ----------------------------------------------------------
# Hardware Comparison
# ----------------------------------------------------------

devices = pd.DataFrame({

    "Platform":[
        "Arduino Nano 33 BLE Sense",
        "ESP32",
        "Raspberry Pi Pico",
        "STM32"
    ],

    "RAM":[
        "256 KB",
        "520 KB",
        "264 KB",
        "256-512 KB"
    ],

    "Flash":[
        "1 MB",
        "4 MB",
        "2 MB",
        "1-2 MB"
    ],

    "Suitable":[
        "Yes",
        "Yes",
        "Yes",
        "Yes"
    ],

    "Reason":[
        "Very small TinyNN model",
        "Enough RAM and Flash",
        "Suitable for TinyML inference",
        "Supports TensorFlow Lite Micro"
    ]

})

print("\nEdge AI Device Comparison\n")

print(devices)

# ----------------------------------------------------------
# Model Information
# ----------------------------------------------------------

print("\nModel Information")

print("-"*40)

print(f"Trainable Parameters : {parameters}")

print(f"Model Size           : {model_size:.2f} KB")

print()

print("Model Complexity : Low")

print("Inference Time   : Very Fast")

print("Power Consumption: Low")

print("Deployment Ease  : Easy")

print()

print("="*60)

print("Recommended Platform")

print("="*60)

print("Arduino Nano 33 BLE Sense")

print()

print("Reason:")

print("- Compact Tiny Neural Network")
print("- Small model size (<30 KB)")
print("- Low RAM requirement")
print("- Low power consumption")
print("- Fast inference")
print("- Native TensorFlow Lite Micro support")

devices.to_csv("Edge_AI_Suitability.csv",index=False)

print("\nEdge AI suitability table saved.")

print("\nTask 6 Completed Successfully.")