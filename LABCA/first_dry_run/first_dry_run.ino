#include <Wire.h>

#define SDA_PIN 21
#define SCL_PIN 22

#define MPU_ADDR 0x68

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_ADDR 0x3C

// MPU6050 registers
#define PWR_MGMT_1 0x6B
#define ACCEL_XOUT_H 0x3B

// ---------- OLED ----------
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

Adafruit_SSD1306 display(
  SCREEN_WIDTH,
  SCREEN_HEIGHT,
  &Wire,
  -1
);

// ---------- Sensor data ----------
struct SensorData {
  float ax;
  float ay;
  float az;
  float gx;
  float gy;
  float gz;
};

// Read one 16-bit signed register value
int16_t read16(uint8_t reg) {

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);

  Wire.requestFrom(MPU_ADDR, (uint8_t)2);

  if (Wire.available() < 2) {
    return 0;
  }

  uint8_t high = Wire.read();
  uint8_t low  = Wire.read();

  return (int16_t)((high << 8) | low);
}

// Read all six sensor values
SensorData readSensor() {

  SensorData data;

  int16_t rawAx = read16(0x3B);
  int16_t rawAy = read16(0x3D);
  int16_t rawAz = read16(0x3F);

  int16_t rawGx = read16(0x43);
  int16_t rawGy = read16(0x45);
  int16_t rawGz = read16(0x47);

  // Default MPU6050 ranges:
  // Accelerometer: ±2g
  // Gyroscope: ±250 °/s

  data.ax = rawAx / 16384.0;
  data.ay = rawAy / 16384.0;
  data.az = rawAz / 16384.0;

  data.gx = rawGx / 131.0;
  data.gy = rawGy / 131.0;
  data.gz = rawGz / 131.0;

  return data;
}

void writeRegister(uint8_t reg, uint8_t value) {

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

void setup() {

  Serial.begin(115200);

  Wire.begin(SDA_PIN, SCL_PIN);

  delay(100);

  // Wake MPU6050
  writeRegister(PWR_MGMT_1, 0x00);

  delay(100);

  // Initialize OLED
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("OLED initialization failed!");
    while (true);
  }

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(2);
  display.setCursor(0, 0);
  display.println("EDGE AI");

  display.setTextSize(1);
  display.setCursor(0, 22);
  display.println("MPU6050 ONLINE");

  display.setCursor(0, 34);
  display.println("SENSOR READY");

  display.display();

  delay(1500);
}

void loop() {

  SensorData data = readSensor();

  // ---------- Serial output ----------

  Serial.print("ACC: ");
  Serial.print(data.ax, 2);
  Serial.print("  ");
  Serial.print(data.ay, 2);
  Serial.print("  ");
  Serial.print(data.az, 2);

  Serial.print("   GYRO: ");
  Serial.print(data.gx, 1);
  Serial.print("  ");
  Serial.print(data.gy, 1);
  Serial.print("  ");
  Serial.println(data.gz, 1);


  // ---------- OLED ----------

  display.clearDisplay();

  display.setTextSize(1);

  display.setCursor(0, 0);
  display.println("EDGE AI SENSOR");

  display.setCursor(0, 12);
  display.print("AX ");
  display.print(data.ax, 2);
  display.print("g");

  display.setCursor(64, 12);
  display.print("AY ");
  display.print(data.ay, 2);
  display.print("g");

  display.setCursor(0, 24);
  display.print("AZ ");
  display.print(data.az, 2);
  display.print("g");

  display.setCursor(0, 38);
  display.print("GX ");
  display.print(data.gx, 0);

  display.setCursor(64, 38);
  display.print("GY ");
  display.print(data.gy, 0);

  display.setCursor(0, 51);
  display.print("GZ ");
  display.print(data.gz, 0);

  display.display();

  delay(100);
}