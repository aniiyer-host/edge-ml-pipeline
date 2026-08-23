import numpy as np

X = np.load("X_test.npy")
y = np.load("y_test.npy")

with open("tflite-micro/tflm_test/test_data.h", "w") as f:
    f.write("#pragma once\n\n")
    f.write("static const float test_input[4] = {\n")
    for value in X[0]:
        f.write(f"    {value:.10f}f,\n")
    f.write("};\n\n")
    f.write(f"static const int test_label = {int(y[0])};\n")

print("Generated test_data.h")
print("Input:", X[0])
print("Actual label:", y[0])