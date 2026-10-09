#include <Wire.h>

#define SDA_PIN 21
#define SCL_PIN 22
#define MPU_ADDR 0x68

void writeRegister(byte reg, byte value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

int16_t read16(byte reg) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);

  Wire.requestFrom(MPU_ADDR, (uint8_t)2);

  if (Wire.available() < 2) {
    return 0;
  }

  byte high = Wire.read();
  byte low = Wire.read();

  return (int16_t)((high << 8) | low);
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Wire.begin(SDA_PIN, SCL_PIN);

  Serial.println();
  Serial.println("=== MPU6050 RAW SENSOR TEST ===");

  // Wake sensor
  writeRegister(0x6B, 0x00);

  delay(100);

  Serial.println("Sensor awakened.");
}

void loop() {
  int16_t ax = read16(0x3B);
  int16_t ay = read16(0x3D);
  int16_t az = read16(0x3F);

  int16_t gx = read16(0x43);
  int16_t gy = read16(0x45);
  int16_t gz = read16(0x47);

  Serial.print("ACC: ");
  Serial.print(ax);
  Serial.print("  ");
  Serial.print(ay);
  Serial.print("  ");
  Serial.print(az);

  Serial.print("   GYRO: ");
  Serial.print(gx);
  Serial.print("  ");
  Serial.print(gy);
  Serial.print("  ");
  Serial.println(gz);

  delay(500);
}