
#include <Arduino.h>
#include <Wire.h>
#include <math.h>

#include <Chirale_TensorFlowLite.h>
#include "gesture_model_int8.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ============================================================
// MPU6050 CONFIGURATION
// ============================================================

#define MPU_ADDR       0x68
#define SDA_PIN        21
#define SCL_PIN        22

#define PWR_MGMT_1     0x6B
#define ACCEL_CONFIG   0x1C
#define GYRO_CONFIG    0x1B
#define ACCEL_XOUT_H   0x3B
#define WHO_AM_I       0x75

#define ACCEL_SCALE    16384.0f
#define GYRO_SCALE     32.8f

// ============================================================
// MODEL CONFIGURATION
// ============================================================

#define WINDOW_SIZE       50
#define NUM_CHANNELS      6
#define NUM_CLASSES       6

#define SAMPLE_RATE_HZ    50
#define SAMPLE_INTERVAL_US (1000000UL / SAMPLE_RATE_HZ)

#define INPUT_SCALE       0.050871994346380234f
#define INPUT_ZERO_POINT  (-19)

// ============================================================
// SCREEN CONFIGURATION
// ============================================================

#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define OLED_ADDRESS 0x3C

Adafruit_SSD1306 display(
  SCREEN_WIDTH,
  SCREEN_HEIGHT,
  &Wire,
  OLED_RESET
);

// Keep the arena size that compiled successfully before.
constexpr size_t TENSOR_ARENA_SIZE = 80 * 1024;
alignas(16) static uint8_t tensorArena[TENSOR_ARENA_SIZE];

const char* gestureNames[NUM_CLASSES] = {
  "idle",
  "wave_left",
  "wave_right",
  "flick_up",
  "flick_down",
  "wrist_rotate"
};

// Training-set scaler statistics.
// Channel order: ax, ay, az, gx, gy, gz.
const float channelMean[NUM_CHANNELS] = {
  -0.06165431f,
   0.03080895f,
   0.97413130f,
  -3.4748738f,
   1.7455415f,
   4.9294290f
};

const float channelStd[NUM_CHANNELS] = {
  0.39220917f,
  0.28669277f,
  0.1887154f,
  47.620644f,
  63.307922f,
  71.37996f
};

// ============================================================
// TENSORFLOW LITE MICRO
// ============================================================

static tflite::AllOpsResolver resolver;

static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;

static TfLiteTensor* inputTensor = nullptr;
static TfLiteTensor* outputTensor = nullptr;

// ============================================================
// SENSOR DATA
// ============================================================

struct IMUData {
  float ax;
  float ay;
  float az;
  float gx;
  float gy;
  float gz;
};

float windowData[WINDOW_SIZE][NUM_CHANNELS];

bool recognitionRunning = false;

// ============================================================
// MPU6050 I2C HELPERS
// ============================================================

bool writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool readRegister(uint8_t reg, uint8_t &value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);

  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  if (Wire.requestFrom(MPU_ADDR, (uint8_t)1) != 1) {
    return false;
  }

  value = Wire.read();
  return true;
}

int16_t readSigned16() {
  uint16_t high = (uint8_t)Wire.read();
  uint16_t low  = (uint8_t)Wire.read();

  return (int16_t)((high << 8) | low);
}

bool initMPU6050() {
  Wire.beginTransmission(MPU_ADDR);

  if (Wire.endTransmission() != 0) {
    Serial.println("ERROR: MPU6050 not responding");
    return false;
  }

  uint8_t identity = 0;

  if (!readRegister(WHO_AM_I, identity)) {
    Serial.println("ERROR: Cannot read WHO_AM_I");
    return false;
  }

  Serial.printf("WHO_AM_I = 0x%02X\n", identity);

  if (!writeRegister(PWR_MGMT_1, 0x00)) {
    return false;
  }

  delay(100);

  if (!writeRegister(ACCEL_CONFIG, 0x00)) {
    return false;
  }

  if (!writeRegister(GYRO_CONFIG, 0x10)) {
    return false;
  }

  delay(100);

  uint8_t accelConfig = 0;
  uint8_t gyroConfig = 0;

  if (!readRegister(ACCEL_CONFIG, accelConfig) ||
      !readRegister(GYRO_CONFIG, gyroConfig)) {
    Serial.println("ERROR: Cannot verify sensor configuration");
    return false;
  }

  Serial.printf("ACCEL_CONFIG = 0x%02X\n", accelConfig);
  Serial.printf("GYRO_CONFIG  = 0x%02X\n", gyroConfig);

  if ((accelConfig & 0x18) != 0x00 ||
      (gyroConfig & 0x18) != 0x10) {
    Serial.println("ERROR: Unexpected sensor range");
    return false;
  }

  Serial.println("MPU6050 initialized");
  return true;
}

bool readMPU6050(IMUData &data) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(ACCEL_XOUT_H);

  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  if (Wire.requestFrom(MPU_ADDR, (uint8_t)14) != 14) {
    return false;
  }

  int16_t axRaw = readSigned16();
  int16_t ayRaw = readSigned16();
  int16_t azRaw = readSigned16();

  // Skip temperature registers.
  Wire.read();
  Wire.read();

  int16_t gxRaw = readSigned16();
  int16_t gyRaw = readSigned16();
  int16_t gzRaw = readSigned16();

  // Physical units must match the training dataset.
  data.ax = (float)axRaw / ACCEL_SCALE;
  data.ay = (float)ayRaw / ACCEL_SCALE;
  data.az = (float)azRaw / ACCEL_SCALE;

  data.gx = (float)gxRaw / GYRO_SCALE;
  data.gy = (float)gyRaw / GYRO_SCALE;
  data.gz = (float)gzRaw / GYRO_SCALE;

  return true;
}

// ============================================================
// TENSORFLOW LITE INITIALIZATION
// ============================================================

bool initModel() {
  model = tflite::GetModel(gesture_model_int8);

  if (model == nullptr) {
    Serial.println("ERROR: Model pointer is null");
    return false;
  }

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Model schema version mismatch");
    return false;
  }

  // Static interpreter avoids heap allocation for the interpreter.
  static tflite::MicroInterpreter staticInterpreter(
    model,
    resolver,
    tensorArena,
    TENSOR_ARENA_SIZE
  );

  interpreter = &staticInterpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors failed");
    return false;
  }

  inputTensor = interpreter->input(0);
  outputTensor = interpreter->output(0);

  if (inputTensor == nullptr || outputTensor == nullptr) {
    Serial.println("ERROR: Missing input/output tensor");
    return false;
  }

  if (inputTensor->type != kTfLiteInt8 ||
      outputTensor->type != kTfLiteInt8) {
    Serial.println("ERROR: Expected INT8 input and output");
    return false;
  }

  if (inputTensor->dims->data[0] != 1 ||
      inputTensor->dims->data[1] != WINDOW_SIZE ||
      inputTensor->dims->data[2] != NUM_CHANNELS) {
    Serial.println("ERROR: Unexpected input dimensions");
    return false;
  }

  if (outputTensor->dims->data[0] != 1 ||
      outputTensor->dims->data[1] != NUM_CLASSES) {
    Serial.println("ERROR: Unexpected output dimensions");
    return false;
  }

  if (fabsf(inputTensor->params.scale - INPUT_SCALE) > 1e-6f ||
      inputTensor->params.zero_point != INPUT_ZERO_POINT) {
    Serial.println("ERROR: Input quantization mismatch");
    Serial.printf("Actual scale: %.9f, zero point: %d\n",
                  inputTensor->params.scale,
                  inputTensor->params.zero_point);
    return false;
  }

  Serial.println("TensorFlow Lite Micro initialized");
  Serial.printf("Input: [1, %d, %d], INT8\n",
                WINDOW_SIZE, NUM_CHANNELS);
  Serial.printf("Input scale: %.9f, zero point: %d\n",
                inputTensor->params.scale,
                inputTensor->params.zero_point);
  Serial.printf("Tensor arena: %u bytes\n",
                (unsigned)TENSOR_ARENA_SIZE);

  return true;
}

// ============================================================
// COLLECT ONE WINDOW AT 50 HZ
// ============================================================

bool collectWindow() {
  unsigned long nextSampleUs = micros();

  for (int sample = 0; sample < WINDOW_SIZE; sample++) {
    // Wait until the next scheduled sample.
    while ((int32_t)(micros() - nextSampleUs) < 0) {
      // Do not print or run inference during acquisition.
    }

    IMUData imu;

    if (!readMPU6050(imu)) {
      Serial.println("ERROR: Sensor read failed; window discarded");
      return false;
    }

    windowData[sample][0] = imu.ax;
    windowData[sample][1] = imu.ay;
    windowData[sample][2] = imu.az;
    windowData[sample][3] = imu.gx;
    windowData[sample][4] = imu.gy;
    windowData[sample][5] = imu.gz;

    nextSampleUs += SAMPLE_INTERVAL_US;
  }

  return true;
}

// ============================================================
// NORMALIZE AND QUANTIZE INPUT
// ============================================================

bool prepareInputTensor() {
  if (inputTensor == nullptr ||
      inputTensor->data.int8 == nullptr) {
    return false;
  }

  for (int sample = 0; sample < WINDOW_SIZE; sample++) {
    for (int channel = 0; channel < NUM_CHANNELS; channel++) {
      float physicalValue = windowData[sample][channel];

      float normalized =
        (physicalValue - channelMean[channel]) /
        channelStd[channel];

      float quantized =
        normalized / INPUT_SCALE + INPUT_ZERO_POINT;

      // Round to nearest integer and clamp to signed INT8.
      long q = lroundf(quantized);

      if (q < -128) q = -128;
      if (q > 127)  q = 127;

      int index = sample * NUM_CHANNELS + channel;
      inputTensor->data.int8[index] = (int8_t)q;
    }
  }

  return true;
}

// ============================================================
// RUN INFERENCE
// ============================================================

bool runInference() {
  if (!prepareInputTensor()) {
    Serial.println("ERROR: Could not prepare input tensor");
    return false;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Model Invoke failed");
    return false;
  }

  int bestClass = 0;
  int bestScore = -129;

  Serial.println();
  Serial.println("---- GESTURE PREDICTION ----");

  for (int i = 0; i < NUM_CLASSES; i++) {
    int score = (int)outputTensor->data.int8[i];

    // All classes share the same output scale and zero point.
    float scoreValue =
      (score - outputTensor->params.zero_point) *
      outputTensor->params.scale;

    Serial.printf("  %-14s score=%d, value=%.4f\n",
                  gestureNames[i], score, scoreValue);

    if (score > bestScore) {
      bestScore = score;
      bestClass = i;
    }
  }

  float confidence =
    (bestScore - outputTensor->params.zero_point) *
    outputTensor->params.scale;

  showGestureOnOLED(gestureNames[bestClass], confidence);
  Serial.printf("\nPREDICTION: %s\n", gestureNames[bestClass]);
  Serial.printf("Quantized output value: %d\n", bestScore);
  Serial.printf("Output score: %.2f%%\n", confidence * 100.0f);
  Serial.println("----------------------------");

  return true;
}

// ============================================================
// SERIAL COMMANDS
// ============================================================

void printMenu() {
  Serial.println();
  Serial.println("====================================");
  Serial.println(" ESP32 TINYML GESTURE RECOGNIZER");
  Serial.println("====================================");
  Serial.println("s = Start continuous recognition");
  Serial.println("x = Stop recognition");
  Serial.println("h = Show help");
  Serial.println();
}

void handleSerial() {
  while (Serial.available() > 0) {
    char command = Serial.read();

    if (command == '\r' || command == '\n') {
      continue;
    }

    if (command == 's' || command == 'S') {
      recognitionRunning = true;
      Serial.println("Recognition enabled");
    } else if (command == 'x' || command == 'X') {
      recognitionRunning = false;
      Serial.println("Recognition stopped");
    } else if (command == 'h' || command == 'H') {
      printMenu();
    }
  }
}

// ============================================================
// OLED HELPERS
// ============================================================

void showOLEDStatus(const char* line1, const char* line2 = "") {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("TINYML GESTURE AI");
  display.drawLine(0, 12, 127, 12, SSD1306_WHITE);

  display.setTextSize(1);
  display.setCursor(0, 24);
  display.println(line1);

  display.setCursor(0, 40);
  display.println(line2);

  display.display();
}

void showGestureOnOLED(const char* gesture, float confidence) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("TINYML GESTURE AI");
  display.drawLine(0, 12, 127, 12, SSD1306_WHITE);

  display.setTextSize(2);
  display.setCursor(0, 23);
  display.println(gesture);

  display.setTextSize(1);
  display.setCursor(0, 49);
  display.print("Score: ");
  display.print(confidence * 100.0f, 1);
  display.println("%");

  display.display();
}

// ============================================================
// SETUP
// ============================================================

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("Starting GestureRecognizerESP32...");

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);


if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS)) {
  Serial.println("ERROR: OLED initialization failed");
  while (true) delay(1000);
}

showOLEDStatus("Initializing...", "Please wait");
Serial.println("OLED initialized");

  if (!initMPU6050()) {
    Serial.println("FATAL: Sensor initialization failed");
    while (true) delay(1000);
  }

  if (!initModel()) {
    Serial.println("FATAL: Model initialization failed");
    while (true) delay(1000);
  }

  showOLEDStatus("System online", "Send s to start");

  Serial.printf("Free heap after initialization: %u bytes\n",
                (unsigned)ESP.getFreeHeap());

  printMenu();
}

// ============================================================
// MAIN LOOP
// ============================================================

void loop() {
  handleSerial();

  if (!recognitionRunning) {
    delay(1);
    return;
  }

  Serial.println("\nCollecting 50 samples...");

  if (!collectWindow()) {
    // Avoid using a partially collected window.
    delay(100);
    return;
  }

  Serial.println("Window captured; running inference...");

  runInference();

  // The next window begins after inference completes.
}