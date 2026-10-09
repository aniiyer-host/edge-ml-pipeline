#include <Wire.h>
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

void setup() {
  Serial.begin(115200);

  Wire.begin(21, 22);

  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS)) {
    Serial.println("OLED initialization failed!");
    while (true);
  }

  Serial.println("OLED initialized!");

  display.clearDisplay();

  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);

  display.setCursor(0, 0);
  display.println("EDGE AI");

  display.setTextSize(1);
  display.setCursor(0, 30);
  display.println("ESP32 + MPU6050");

  display.setCursor(0, 45);
  display.println("SYSTEM ONLINE");

  display.display();
}

void loop() {
}