#include <Arduino.h>
#include <Wire.h>

// ============================================================
// MPU6050 CONFIGURATION
// ============================================================

#define MPU_ADDR       0x68

#define PWR_MGMT_1     0x6B
#define ACCEL_CONFIG   0x1C
#define GYRO_CONFIG    0x1B
#define ACCEL_XOUT_H   0x3B
#define WHO_AM_I       0x75

// ESP32 I2C pins
#define SDA_PIN        21
#define SCL_PIN        22

// MPU6050 scale factors
// ±2g       -> 16384 LSB/g
// ±1000 dps  -> 32.8 LSB/(deg/s)
#define ACCEL_SCALE    16384.0f
#define GYRO_SCALE     32.8f

// ============================================================
// DATA COLLECTION CONFIGURATION
// ============================================================

#define SAMPLE_RATE_HZ       50
#define SAMPLE_INTERVAL_US   (1000000UL / SAMPLE_RATE_HZ)

#define WINDOW_SIZE           50

// ============================================================
// GESTURE DEFINITIONS
// ============================================================

const char* gestureNames[] = {
  "idle",
  "wave_left",
  "wave_right",
  "flick_up",
  "flick_down",
  "wrist_rotate"
};

const int NUM_GESTURES = 6;

// ============================================================
// STATE
// ============================================================

bool collecting = false;

int selectedGesture = -1;
unsigned long windowNumber = 0;

int sampleCount = 0;

unsigned long windowStartMs = 0;
unsigned long nextSampleUs = 0;

// ============================================================
// SENSOR DATA STRUCTURE
// ============================================================

struct IMUData {
  float ax;
  float ay;
  float az;
  float gx;
  float gy;
  float gz;
};

// ============================================================
// I2C HELPERS
// ============================================================

void writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

uint8_t readRegister(uint8_t reg) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);

  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)1, (uint8_t)true);

  if (Wire.available()) {
    return Wire.read();
  }

  return 0xFF;
}

// ============================================================
// MPU6050 INITIALIZATION
// ============================================================

bool initMPU6050() {

  // Check whether the device responds
  Wire.beginTransmission(MPU_ADDR);

  if (Wire.endTransmission() != 0) {
    Serial.println("# ERROR: MPU6050 not responding at 0x68");
    return false;
  }

  uint8_t whoAmI = readRegister(WHO_AM_I);

  Serial.print("# WHO_AM_I = 0x");

  if (whoAmI < 0x10) {
    Serial.print("0");
  }

  Serial.println(whoAmI, HEX);

  // Wake up MPU6050
  writeRegister(PWR_MGMT_1, 0x00);

  delay(100);

  // Accelerometer ±2g
  writeRegister(ACCEL_CONFIG, 0x00);

  // Gyroscope ±1000 deg/s
  writeRegister(GYRO_CONFIG, 0x10);

  delay(100);

  Serial.println("# MPU6050 initialized");

  return true;
}

// ============================================================
// READ MPU6050
// ============================================================

bool readMPU6050(IMUData &data) {

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(ACCEL_XOUT_H);

  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  // MPU6050 gives:
  //
  // AX high
  // AX low
  // AY high
  // AY low
  // AZ high
  // AZ low
  // Temperature high
  // Temperature low
  // GX high
  // GX low
  // GY high
  // GY low
  // GZ high
  // GZ low

  uint8_t bytesRequested = 14;

  uint8_t bytesReceived =
      Wire.requestFrom(
        (uint8_t)MPU_ADDR,
        bytesRequested,
        (uint8_t)true
      );

  if (bytesReceived != 14) {
    return false;
  }

  int16_t axRaw =
      ((int16_t)Wire.read() << 8) | Wire.read();

  int16_t ayRaw =
      ((int16_t)Wire.read() << 8) | Wire.read();

  int16_t azRaw =
      ((int16_t)Wire.read() << 8) | Wire.read();

  // Skip temperature
  Wire.read();
  Wire.read();

  int16_t gxRaw =
      ((int16_t)Wire.read() << 8) | Wire.read();

  int16_t gyRaw =
      ((int16_t)Wire.read() << 8) | Wire.read();

  int16_t gzRaw =
      ((int16_t)Wire.read() << 8) | Wire.read();

  // Convert to physical units
  data.ax = (float)axRaw / ACCEL_SCALE;
  data.ay = (float)ayRaw / ACCEL_SCALE;
  data.az = (float)azRaw / ACCEL_SCALE;

  data.gx = (float)gxRaw / GYRO_SCALE;
  data.gy = (float)gyRaw / GYRO_SCALE;
  data.gz = (float)gzRaw / GYRO_SCALE;

  return true;
}

// ============================================================
// PRINT CSV HEADER
// ============================================================

void printCSVHeader() {
  
  Serial.println(
    "label,sample_id,timestamp_ms,sample,"
    "ax,ay,az,gx,gy,gz"
  );
}

// ============================================================
// START COLLECTION
// ============================================================

void startCollection() {

  if (selectedGesture < 0 ||
      selectedGesture >= NUM_GESTURES) {

    Serial.println("# ERROR: Select a gesture first");

    return;
  }

  collecting = true;

  sampleCount = 0;

  windowStartMs = millis();

  nextSampleUs = micros();

  windowNumber++;

  Serial.println();
  Serial.println("# BEGIN_WINDOW");

  Serial.print("# gesture = ");
  Serial.println(gestureNames[selectedGesture]);

  Serial.print("# window = ");
  Serial.println(windowNumber);

  Serial.print("# samples = ");
  Serial.println(WINDOW_SIZE);

  Serial.print("# sample_rate_hz = ");
  Serial.println(SAMPLE_RATE_HZ);

  Serial.println("# ----------------------------------------");

  printCSVHeader();
}

// ============================================================
// END COLLECTION
// ============================================================

void endCollection() {

  collecting = false;

  Serial.println("# ----------------------------------------");
  Serial.println("# END_WINDOW");

  Serial.print("# gesture = ");
  Serial.println(gestureNames[selectedGesture]);

  Serial.print("# window = ");
  Serial.println(windowNumber);

  Serial.print("# samples_collected = ");
  Serial.println(sampleCount);

  Serial.println("# READY");
  Serial.println();
}

// ============================================================
// CANCEL COLLECTION
// ============================================================

void cancelCollection() {

  collecting = false;

  Serial.println();
  Serial.println("# WINDOW_DISCARDED");
  Serial.println("# READY");
  Serial.println();
}

// ============================================================
// SERIAL MENU
// ============================================================

void printMenu() {

  Serial.println();
  Serial.println("==========================================");
  Serial.println(" ESP32 GESTURE DATA COLLECTOR");
  Serial.println("==========================================");

  Serial.println();
  Serial.println("Select gesture:");

  for (int i = 0; i < NUM_GESTURES; i++) {

    Serial.print("  ");
    Serial.print(i + 1);
    Serial.print(" = ");
    Serial.println(gestureNames[i]);
  }

  Serial.println();
  Serial.println("Commands:");
  Serial.println("  1-6 = Select gesture");
  Serial.println("  s   = Start collection");
  Serial.println("  x   = Cancel current window");
  Serial.println("  r   = Reset window counter");
  Serial.println("  h   = Show this menu");

  Serial.println();
  Serial.println("# READY");
}

// ============================================================
// HANDLE SERIAL INPUT
// ============================================================

void handleSerial() {

  if (!Serial.available()) {
    return;
  }

  char command = Serial.read();

  // Ignore newline / carriage return
  if (command == '\n' || command == '\r') {
    return;
  }

  // ----------------------------------------------------------
  // Gesture selection
  // ----------------------------------------------------------

  if (command >= '1' &&
      command <= '6') {

    if (collecting) {
      Serial.println(
        "# ERROR: Cannot change gesture while collecting"
      );

      return;
    }

    selectedGesture = command - '1';

    Serial.print("# Selected gesture: ");
    Serial.println(
      gestureNames[selectedGesture]
    );

    return;
  }

  // ----------------------------------------------------------
  // Start
  // ----------------------------------------------------------

  if (command == 's' || command == 'S') {

    if (collecting) {
      Serial.println("# ERROR: Already collecting");
      return;
    }

    if (selectedGesture < 0) {

      Serial.println(
        "# ERROR: Select a gesture first (1-6)"
      );

      return;
    }

    startCollection();

    return;
  }

  // ----------------------------------------------------------
  // Cancel
  // ----------------------------------------------------------

  if (command == 'x' || command == 'X') {

    if (collecting) {
      cancelCollection();
    }
    else {
      Serial.println("# Nothing to cancel");
    }

    return;
  }

  // ----------------------------------------------------------
  // Reset window counter
  // ----------------------------------------------------------

  if (command == 'r' || command == 'R') {

    if (collecting) {

      Serial.println(
        "# ERROR: Cannot reset counter while collecting"
      );

      return;
    }

    windowNumber = 0;

    Serial.println("# Window counter reset to 0");

    return;
  }

  // ----------------------------------------------------------
  // Help
  // ----------------------------------------------------------

  if (command == 'h' || command == 'H') {

    printMenu();

    return;
  }
}

// ============================================================
// SETUP
// ============================================================

void setup() {

  Serial.begin(115200);

  delay(1000);

  Serial.println();
  Serial.println("==========================================");
  Serial.println(" ESP32 TOUCHLESS GESTURE DATA COLLECTOR");
  Serial.println("==========================================");

  Serial.println("# Initializing I2C...");

  Wire.begin(SDA_PIN, SCL_PIN);

  // 400 kHz I2C
  Wire.setClock(400000);

  Serial.print("# SDA = GPIO");
  Serial.println(SDA_PIN);

  Serial.print("# SCL = GPIO");
  Serial.println(SCL_PIN);

  Serial.println("# I2C clock = 400 kHz");

  Serial.println();

  // Initialize MPU6050
  if (!initMPU6050()) {

    Serial.println();
    Serial.println("# FATAL: MPU6050 initialization failed");
    Serial.println("# Check wiring and power.");
    Serial.println();

    while (true) {
      delay(1000);
    }
  }

  Serial.println();

  Serial.println("# Configuration:");
  Serial.println("# Sample rate = 50 Hz");
  Serial.println("# Window size = 50 samples");
  Serial.println("# Window duration = 1 second");
  Serial.println("# Axes = 6");
  Serial.println("# Accelerometer range = ±2g");
  Serial.println("# Gyroscope range = ±1000 deg/s");

  printMenu();
}

// ============================================================
// MAIN LOOP
// ============================================================

void loop() {

  // Always allow serial commands
  handleSerial();

  if (!collecting) {
    delay(1);
    return;
  }

  // ----------------------------------------------------------
  // Fixed-rate sampling
  // ----------------------------------------------------------

  unsigned long nowUs = micros();

  if ((long)(nowUs - nextSampleUs) < 0) {
    return;
  }

  // Schedule next sample
  nextSampleUs += SAMPLE_INTERVAL_US;

  // ----------------------------------------------------------
  // Read IMU
  // ----------------------------------------------------------

  IMUData imu;

  if (!readMPU6050(imu)) {

    Serial.println(
      "# ERROR: Failed to read MPU6050"
    );

    cancelCollection();

    return;
  }

  // ----------------------------------------------------------
  // Timestamp
  // ----------------------------------------------------------

  unsigned long timestampMs =
      millis() - windowStartMs;

  // ----------------------------------------------------------
  // Print dataset row
  // ----------------------------------------------------------

  Serial.print(gestureNames[selectedGesture]);
  Serial.print(",");

  Serial.print(windowNumber);
  Serial.print(",");

  Serial.print(sampleCount);
  Serial.print(",");

  Serial.print(timestampMs);
  Serial.print(",");

  Serial.print(imu.ax, 4);
  Serial.print(",");

  Serial.print(imu.ay, 4);
  Serial.print(",");

  Serial.print(imu.az, 4);
  Serial.print(",");

  Serial.print(imu.gx, 4);
  Serial.print(",");

  Serial.print(imu.gy, 4);
  Serial.print(",");

  Serial.println(imu.gz, 4);

  sampleCount++;

  // ----------------------------------------------------------
  // Window complete
  // ----------------------------------------------------------

  if (sampleCount >= WINDOW_SIZE) {

    endCollection();
  }
}
