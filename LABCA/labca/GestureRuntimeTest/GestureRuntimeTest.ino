
#include <Chirale_TensorFlowLite.h>
#include "gesture_model_int8.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr int kTensorArenaSize = 80 * 1024;

alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void stopWithError(const char* message) {
  Serial.print("ERROR: ");
  Serial.println(message);
  while (true) {
    delay(1000);
  }
}

void setup() {
  Serial.begin(115200);
  delay(1500);

  Serial.println();
  Serial.println("=== Gesture INT8 Runtime Test ===");
  Serial.printf("Model bytes: %u\n", gesture_model_int8_len);
  Serial.printf("Free heap before init: %u bytes\n", ESP.getFreeHeap());

  model = tflite::GetModel(gesture_model_int8);

  if (model == nullptr) {
    stopWithError("GetModel returned null");
  }

  Serial.printf("Model schema version: %d\n", model->version());
  Serial.printf("Runtime schema version: %d\n", TFLITE_SCHEMA_VERSION);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    stopWithError("Model/runtime schema mismatch");
  }

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);

  interpreter = &static_interpreter;

  Serial.println("Allocating tensors...");
  TfLiteStatus status = interpreter->AllocateTensors();

  if (status != kTfLiteOk) {
    stopWithError("AllocateTensors failed");
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  if (input == nullptr || output == nullptr) {
    stopWithError("Input/output tensor unavailable");
  }

  Serial.println("Tensor allocation successful!");
  Serial.printf("Free heap after allocation: %u bytes\n", ESP.getFreeHeap());

  Serial.printf("Input type: %d\n", input->type);
  Serial.printf("Input bytes: %u\n", input->bytes);
  Serial.printf("Input scale: %.9f\n", input->params.scale);
  Serial.printf("Input zero point: %d\n", input->params.zero_point);

  Serial.print("Input dimensions: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.printf("%d ", input->dims->data[i]);
  }
  Serial.println();

  Serial.printf("Output type: %d\n", output->type);
  Serial.printf("Output bytes: %u\n", output->bytes);
  Serial.printf("Output scale: %.9f\n", output->params.scale);
  Serial.printf("Output zero point: %d\n", output->params.zero_point);

  Serial.print("Output dimensions: ");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.printf("%d ", output->dims->data[i]);
  }
  Serial.println();

  if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8) {
    stopWithError("Expected INT8 input and output tensors");
  }

  if (input->dims->size != 3 ||
      input->dims->data[0] != 1 ||
      input->dims->data[1] != 50 ||
      input->dims->data[2] != 6) {
    stopWithError("Unexpected input shape; expected [1, 50, 6]");
  }

  if (output->dims->size != 2 ||
      output->dims->data[0] != 1 ||
      output->dims->data[1] != 6) {
    stopWithError("Unexpected output shape; expected [1, 6]");
  }

  // Zero-filled normalized input window, quantized for the model.
  // q = round(0 / scale + zero_point) = zero_point.
  for (int i = 0; i < 50 * 6; i++) {
    input->data.int8[i] = (int8_t)input->params.zero_point;
  }

  Serial.println("Running one inference...");
  status = interpreter->Invoke();

  if (status != kTfLiteOk) {
  stopWithError("Initial Invoke failed");
  }

  constexpr int kBenchmarkRuns = 100;

  // Warm up the interpreter before measuring.
  for (int i = 0; i < 5; i++) {
    if (interpreter->Invoke() != kTfLiteOk) {
      stopWithError("Warm-up inference failed");
    }
  }

  uint32_t start_us = micros();

  for (int i = 0; i < kBenchmarkRuns; i++) {
    if (interpreter->Invoke() != kTfLiteOk) {
      stopWithError("Benchmark inference failed");
    }
  }

  uint32_t elapsed_us = micros() - start_us;

  Serial.printf("Benchmark runs: %d\n", kBenchmarkRuns);
  Serial.printf("Total inference time: %lu us\n",
                (unsigned long)elapsed_us);
  Serial.printf("Average inference time: %.2f ms\n",
                elapsed_us / (1000.0f * kBenchmarkRuns));
  Serial.printf("Free heap after benchmark: %u bytes\n",
              ESP.getFreeHeap());

  if (status != kTfLiteOk) {
    stopWithError("Invoke failed");
  }

  const char* labels[] = {
    "idle", "wave_left", "wave_right",
    "flick_up", "flick_down", "wrist_rotate"
  };

  int best_index = 0;

  Serial.println("INT8 output scores:");
  for (int i = 0; i < 6; i++) {
    Serial.printf("  %s: %d\n", labels[i], output->data.int8[i]);

    if (output->data.int8[i] > output->data.int8[best_index]) {
      best_index = i;
    }
  }

  Serial.printf("Predicted class: %s (index %d)\n",
                labels[best_index], best_index);

  Serial.println("=== RUNTIME TEST PASSED ===");
}

void loop() {
  delay(1000);
}