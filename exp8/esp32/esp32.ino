#include <Chirale_TensorFlowLite.h>

#include <math.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "../model_data.h"

// ===============================
// Hardware
// ===============================

const int LDR_PIN = 34;
const int LASER_PIN = 25;

// ===============================
// TensorFlow Lite globals
// ===============================

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor arena
constexpr int kTensorArenaSize = 8 * 1024;

alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// ===============================
// Quantization parameters
// ===============================

constexpr float INPUT_SCALE = 0.003921568859368563f;
constexpr int INPUT_ZERO_POINT = -128;

constexpr float OUTPUT_SCALE = 0.00390625f;
constexpr int OUTPUT_ZERO_POINT = -128;

// ===============================
// Profiling
// ===============================

const int NUM_MEASUREMENTS = 50;

unsigned long totalInferenceTime = 0;
unsigned long minInferenceTime = 999999;
unsigned long maxInferenceTime = 0;

int measurementCount = 0;

// ===============================
// Setup
// ===============================

void setup() {

  Serial.begin(115200);
  delay(1500);

  pinMode(LDR_PIN, INPUT);
  pinMode(LASER_PIN, OUTPUT);

  // Turn laser ON
  digitalWrite(LASER_PIN, HIGH);

  Serial.println();
  Serial.println("========================================");
  Serial.println(" Experiment 8 - Edge AI");
  Serial.println(" ESP32 + TFLM Laser Classifier");
  Serial.println(" Task 5 - Profiling");
  Serial.println("========================================");

  // ===============================
  // Load model
  // ===============================

  model = tflite::GetModel(laser_classifier_int8_tflite);

  if (model->version() != TFLITE_SCHEMA_VERSION) {

    Serial.println("ERROR: Model schema version mismatch!");

    while (true) {
      delay(1000);
    }
  }

  Serial.println("Model loaded successfully.");

  // ===============================
  // Register operators
  // ===============================

  static tflite::MicroMutableOpResolver<3> resolver;

  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddSoftmax();

  // ===============================
  // Create interpreter
  // ===============================

  static tflite::MicroInterpreter static_interpreter(
    model,
    resolver,
    tensor_arena,
    kTensorArenaSize
  );

  interpreter = &static_interpreter;

  // ===============================
  // Allocate tensors
  // ===============================

  if (interpreter->AllocateTensors() != kTfLiteOk) {

    Serial.println("ERROR: AllocateTensors() failed!");

    while (true) {
      delay(1000);
    }
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Tensor allocation successful.");

  Serial.print("Tensor arena allocated: ");
  Serial.print(kTensorArenaSize);
  Serial.println(" bytes");

  Serial.print("Input type: ");
  Serial.println(input->type);

  Serial.print("Output type: ");
  Serial.println(output->type);

  Serial.println();
  Serial.println("========================================");
  Serial.println("Live inference started.");
  Serial.println("========================================");
}

// ===============================
// Main loop
// ===============================

void loop() {

  // ===============================
  // Read sensor
  // ===============================

  int ldrState = digitalRead(LDR_PIN);

  float sensorValue;

  if (ldrState == LOW) {
    // Laser detected
    sensorValue = 0.0f;
  }
  else {
    // No laser
    sensorValue = 1.0f;
  }

  // ===============================
  // Quantize input
  // ===============================

  int8_t quantizedInput =
      static_cast<int8_t>(
        round(sensorValue / INPUT_SCALE)
        + INPUT_ZERO_POINT
      );

  input->data.int8[0] = quantizedInput;

  // ===============================
  // Measure inference
  // ===============================

  unsigned long startTime = micros();

  TfLiteStatus invokeStatus = interpreter->Invoke();

  unsigned long endTime = micros();

  if (invokeStatus != kTfLiteOk) {

    Serial.println("ERROR: Inference failed!");

    delay(500);

    return;
  }

  unsigned long inferenceTime =
      endTime - startTime;

  // ===============================
  // Update profiling statistics
  // ===============================

  totalInferenceTime += inferenceTime;

  if (inferenceTime < minInferenceTime) {
    minInferenceTime = inferenceTime;
  }

  if (inferenceTime > maxInferenceTime) {
    maxInferenceTime = inferenceTime;
  }

  measurementCount++;

  // ===============================
  // Read model output
  // ===============================

  int8_t output0 = output->data.int8[0];
  int8_t output1 = output->data.int8[1];

  float probability0 =
      (output0 - OUTPUT_ZERO_POINT)
      * OUTPUT_SCALE;

  float probability1 =
      (output1 - OUTPUT_ZERO_POINT)
      * OUTPUT_SCALE;

  // Class 0 = NO_LASER
  // Class 1 = LASER_DETECTED

  int predictedClass;

  if (probability0 > probability1) {
    predictedClass = 0;
  }
  else {
    predictedClass = 1;
  }

  // ===============================
  // Actuation / response
  // ===============================

  Serial.print("LDR: ");
  Serial.print(ldrState);

  Serial.print(" | Input INT8: ");
  Serial.print(quantizedInput);

  Serial.print(" | NO_LASER: ");
  Serial.print(probability0, 3);

  Serial.print(" | LASER: ");
  Serial.print(probability1, 3);

  Serial.print(" | Prediction: ");

  if (predictedClass == 1) {
    Serial.print("LASER DETECTED");
  }
  else {
    Serial.print("NO LASER");
  }

  Serial.print(" | Inference: ");
  Serial.print(inferenceTime);
  Serial.println(" us");

  // ===============================
  // Print profiling summary
  // ===============================

  if (measurementCount >= NUM_MEASUREMENTS) {

    unsigned long averageInferenceTime =
        totalInferenceTime / NUM_MEASUREMENTS;

    Serial.println();
    Serial.println("========================================");
    Serial.println(" PROFILING RESULTS");
    Serial.println("========================================");

    Serial.print("Measurements: ");
    Serial.println(NUM_MEASUREMENTS);

    Serial.print("Average inference: ");
    Serial.print(averageInferenceTime);
    Serial.println(" us");

    Serial.print("Minimum inference: ");
    Serial.print(minInferenceTime);
    Serial.println(" us");

    Serial.print("Maximum inference: ");
    Serial.print(maxInferenceTime);
    Serial.println(" us");

    Serial.print("Tensor arena allocated: ");
    Serial.print(kTensorArenaSize);
    Serial.println(" bytes");

    Serial.println("========================================");

    // Reset profiling counters
    measurementCount = 0;
    totalInferenceTime = 0;
    minInferenceTime = 999999;
    maxInferenceTime = 0;
  }

  delay(200);
}