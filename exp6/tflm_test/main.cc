#include <cstdint>
#include <cstdio>
#include <chrono>
#include <cmath>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "test_data.h"

int main() {

    // ---------------------------------------------------------
    // 1. Load TFLite model from C byte array
    // ---------------------------------------------------------
    const tflite::Model* model =
        tflite::GetModel(TinyNN_int8_tflite);

    // Verify model schema version
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema mismatch!\n");
        printf("Model version: %d\n", model->version());
        printf("Expected version: %d\n", TFLITE_SCHEMA_VERSION);
        return 1;
    }

    // ---------------------------------------------------------
    // 2. Register only operators used by the model
    // ---------------------------------------------------------
    tflite::MicroMutableOpResolver<2> resolver;

    if (resolver.AddFullyConnected() != kTfLiteOk) {
        printf("Failed to register FullyConnected\n");
        return 1;
    }

    if (resolver.AddSoftmax() != kTfLiteOk) {
        printf("Failed to register Softmax\n");
        return 1;
    }

    // ---------------------------------------------------------
    // 3. Tensor Arena
    // ---------------------------------------------------------
    constexpr int kTensorArenaSize = 16 * 1024;

    static uint8_t tensor_arena[kTensorArenaSize];

    // ---------------------------------------------------------
    // 4. Create interpreter
    // ---------------------------------------------------------
    tflite::MicroInterpreter interpreter(
        model,
        resolver,
        tensor_arena,
        kTensorArenaSize
    );

    // ---------------------------------------------------------
    // 5. Allocate tensors
    // ---------------------------------------------------------
    TfLiteStatus status = interpreter.AllocateTensors();

    if (status != kTfLiteOk) {
        printf("AllocateTensors() FAILED!\n");
        return 1;
    }

    printf("TFLM initialization successful!\n");
    printf("Tensor Arena Size: %d bytes\n", kTensorArenaSize);
    printf("Tensor Arena Used: %zu bytes\n",interpreter.arena_used_bytes());

    // ---------------------------------------------------------
    // 6. Get input and output tensors
    // ---------------------------------------------------------
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    if (input == nullptr || output == nullptr) {
        printf("Failed to get input/output tensor!\n");
        return 1;
    }

    printf("\nInput tensor:\n");
    printf("  Type: %d\n", input->type);
    printf("  Scale: %f\n", input->params.scale);
    printf("  Zero Point: %d\n", input->params.zero_point);

    printf("\nOutput tensor:\n");
    printf("  Type: %d\n", output->type);
    printf("  Scale: %f\n", output->params.scale);
    printf("  Zero Point: %d\n", output->params.zero_point);

        // ---------------------------------------------------------
    // 7. Run inference on all test samples
    // ---------------------------------------------------------

    if (input->type != kTfLiteInt8) {
        printf("Expected INT8 input tensor!\n");
        return 1;
    }

    if (output->type != kTfLiteInt8) {
        printf("Expected INT8 output tensor!\n");
        return 1;
    }

    int correct_predictions = 0;
    double total_latency_ms = 0.0;

    printf("\n========================================\n");
    printf("Running inference on %d test samples\n", kNumTestSamples);
    printf("========================================\n");

    for (int sample_index = 0;
         sample_index < kNumTestSamples;
         sample_index++) {

        // Get current test sample
        const float* iris_sample = X_test_data[sample_index];

        // ---------------------------------------------------------
        // Quantize FLOAT32 -> INT8
        // ---------------------------------------------------------

        for (int i = 0; i < kNumFeatures; i++) {

            int32_t quantized =
                static_cast<int32_t>(
                    iris_sample[i] / input->params.scale
                ) + input->params.zero_point;

            // Clamp to INT8 range
            if (quantized > 127)
                quantized = 127;

            if (quantized < -128)
                quantized = -128;

            input->data.int8[i] =
                static_cast<int8_t>(quantized);
        }

        // ---------------------------------------------------------
        // Run inference and measure latency
        // ---------------------------------------------------------

        auto start = std::chrono::high_resolution_clock::now();

        status = interpreter.Invoke();

        auto end = std::chrono::high_resolution_clock::now();

        if (status != kTfLiteOk) {
            printf(
                "Inference FAILED for sample %d!\n",
                sample_index
            );
            return 1;
        }

        double latency_ms =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        total_latency_ms += latency_ms;

        // ---------------------------------------------------------
        // Read and dequantize output
        // ---------------------------------------------------------

        int predicted_class = 0;
        float max_probability = -1.0f;

        for (int i = 0; i < 3; i++) {

            int8_t quantized_output =
                output->data.int8[i];

            float probability =
                (static_cast<int>(quantized_output)
                 - output->params.zero_point)
                * output->params.scale;

            if (probability > max_probability) {
                max_probability = probability;
                predicted_class = i;
            }
        }

        // ---------------------------------------------------------
        // Compare prediction with actual label
        // ---------------------------------------------------------

        int actual_class = y_test_data[sample_index];

        bool correct =
            (predicted_class == actual_class);

        if (correct) {
            correct_predictions++;
        }

        // Print sample result
        printf(
            "Sample %2d | Actual: %d | Predicted: %d | "
            "%s | Latency: %.4f ms\n",

            sample_index + 1,
            actual_class,
            predicted_class,
            correct ? "CORRECT" : "INCORRECT",
            latency_ms
        );
    }

    // ---------------------------------------------------------
    // 8. Final evaluation results
    // ---------------------------------------------------------

    double accuracy =
        (static_cast<double>(correct_predictions) /
         kNumTestSamples) * 100.0;

    double average_latency_ms =
        total_latency_ms / kNumTestSamples;

    printf("\n========================================\n");
    printf("FINAL RESULTS\n");
    printf("========================================\n");

    printf("Total test samples: %d\n",
           kNumTestSamples);

    printf("Correct predictions: %d\n",
           correct_predictions);

    printf("Classification Accuracy: %.2f%%\n",
           accuracy);

    printf("Average Inference Latency: %.4f ms\n",
           average_latency_ms);

    printf("Tensor Arena Used: %zu bytes\n",
           interpreter.arena_used_bytes());

    printf("========================================\n");

    return 0;
}