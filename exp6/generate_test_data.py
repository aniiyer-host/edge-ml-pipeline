import numpy as np

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

with open("test_data.h", "w") as f:
    f.write("#ifndef TEST_DATA_H\n")
    f.write("#define TEST_DATA_H\n\n")

    f.write(f"constexpr int kNumTestSamples = {len(X_test)};\n")
    f.write(f"constexpr int kNumFeatures = {X_test.shape[1]};\n\n")

    f.write("const float X_test_data[kNumTestSamples][kNumFeatures] = {\n")

    for sample in X_test:
        values = ", ".join(f"{x:.8f}f" for x in sample)
        f.write(f"    {{{values}}},\n")

    f.write("};\n\n")

    f.write("const int y_test_data[kNumTestSamples] = {\n    ")

    for i, label in enumerate(y_test):
        f.write(str(int(label)))

        if i < len(y_test) - 1:
            f.write(", ")

        if (i + 1) % 10 == 0 and i < len(y_test) - 1:
            f.write("\n    ")

    f.write("\n};\n\n")
    f.write("#endif\n")

print("test_data.h generated successfully!")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)