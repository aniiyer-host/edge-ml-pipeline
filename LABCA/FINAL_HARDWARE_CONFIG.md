# Final Hardware Configuration

Based on the project evolution and documented decisions in `LABCA/Hardware_confs.md` and firmware source code, the final hardware configuration is:

## Core Components

| Component | Specification | Location/Pin |
|-----------|--------------|--------------|
| **Microcontroller** | ESP32-WROOM DevKit | - |
| **IMU Sensor** | MPU6050 (GY-521 breakout) | - |
| **Communication** | I²C Bus | SDA = GPIO21, SCL = GPIO22 |
| **Power** | USB 5V (development) | VBUS pin |
| **Optional Display** | SSD1306 OLED (128x64) | I²C address 0x3C |

## Sensor Configuration

### MPU6050 Settings
| Parameter | Setting | Register | Value | Notes |
|-----------|---------|----------|-------|-------|
| **Accelerometer Range** | ±2g | ACCEL_CONFIG (0x1C) | 0x00 | 000 = ±2g |
| **Gyroscope Range** | ±1000°/s | GYRO_CONFIG (0x1B) | 0x10 | 10000 = ±1000°/s |
| **Clock Source** | Internal 8MHz | PWR_MGMT_1 (0x6B) | 0x00 | 0000000 = Internal oscillator |
| **Sleep Mode** | Disabled | PWR_MGMT_1 (0x6B) | 0x00 | Bit 6 = 0 (not sleeping) |
| **I²C Address** | 0x68 | - | - | AD0 pin connected to GND |

### Scale Factors (Firmware Constants)
| Parameter | Scale Factor | Derivation | Units |
|-----------|--------------|------------|-------|
| **Accelerometer** | 16384.0 | 2^15 / 2g | LSB/g |
| **Gyroscope** | 32.8 | 2^15 / 2000°/s | LSB/(°/s) |

*Note: The gyroscope scale factor of 32.8 LSB/(°/s) corresponds to ±2000°/s full-scale range, but we configured for ±1000°/s, giving effective sensitivity of ~65.6 LSB/(°/s). However, the firmware appears to use 16384.0 for accel and 131.0 for gyro (±250°/s) in early versions, later updated.*

### Sampling Configuration
| Parameter | Value | Derivation |
|-----------|-------|------------|
| **Sampling Frequency** | 50 Hz | Chosen for Nyquist-appropriate human motion capture |
| **Sample Period** | 20 ms | 1 / 50 Hz |
| **Window Size** | 50 samples | Fixed window for processing |
| **Window Duration** | 1.0 second | 50 samples × 20 ms/sample |
| **Data Rate** | 300 values/second | 50 Hz × 6 channels |

## Wiring Diagram

```
MPU6050       ESP32-WROOM
--------      ------------
VCC           3V3 Pin
GND           GND Pin
SCL           GPIO22
SDA           GPIO21
AD0           GND (sets I²C addr to 0x68)
INT           Not Connected

Optional (OLED Display):
SSD1306 VCC   3V3 Pin
SSD1306 GND   GND Pin
SSD1306 SCL   GPIO22 (shared with MPU6050)
SSD1306 SDA   GPIO21 (shared with MPU6050)
```

## Firmware Pin Definitions (from first_dry_run.ino and LABCA configs)
```cpp
#define SDA_PIN 21
#define SCL_PIN 22
#define MPU_ADDR 0x68
#define PWR_MGMT_1 0x6B
#define ACCEL_XOUT_H 0x3B
```

## Validation and Testing

### Communication Verification
1. **I²C Scanner**: Successfully detected device at address 0x68
2. **WHO_AM_I Register**: Returned value 0x70 (unexpected but valid for communication)
   - Important: 0x68 is the I²C address used for communication
   - 0x70 is the WHO_AM_I register value (device identification)
   - These are DIFFERENT values serving different purposes
   - Do NOT confuse the I²C address (0x68) with the WHO_AM_I value (0x70)

### Sensor Response Validation
- **WHO_AM_I** (Register 0x75): Read value 0x70 confirms device responsiveness
- **PWR_MGMT_1** (Register 0x6B): Written value 0x00 to wake from sleep mode
- **ACCEL_CONFIG** (Register 0x1C): Written value 0x00 to set ±2g range
- **GYRO_CONFIG** (Register 0x1B): Written value 0x10 to set ±1000°/s range

## Rationale for Final Choices

### Why ESP32-WROOM?
- Successor to ESP8266 used in early prototypes
- Dual-core processor provides headroom for feature extraction
- Integrated WiFi for wireless data transmission
- Sufficient RAM (520KB) and flash (4MB) for ML applications
- Arduino IDE compatibility via ESP32 core

### Why MPU6050?
- 6-axis IMU (3-axis accelerometer + 3-axis gyroscope) provides motion dynamics
- Well-established, low-cost sensor with extensive library support
- I²C interface simplifies wiring (only 2 signal lines + power)
- Adequate performance for human gesture recognition tasks

### Why I²C Communication?
- Minimal pin usage (only SDA and SCL required)
- Adequate bandwidth for 50 Hz × 6 channels = 300 values/second
- Built-in Wire library in Arduino/ESP32 ecosystems
- Simple master-slave protocol suitable for sensor communication

### Why ±2g Accelerometer Range?
- Typical human hand motions produce accelerations well under 2g
- Provides sufficient resolution (~0.0001g per LSB) for gesture discrimination
- Avoids unnecessary noise from higher ranges (±4g, ±8g, ±16g)

### Why ±1000°/s Gyroscope Range?
- Empirical testing showed ±250°/s and ±500°/s ranges saturated during wrist rotation
- Saturation causes clipping and loss of dynamic information critical for ML
- ±1000°/s range provided readings up to ~±822°/s during testing without clipping
- Balances range needs with resolution (approximately 0.03°/s per LSB at this range)

### Why 50 Hz Sampling?
- Human gesture movements primarily occur below 25 Hz
- Nyquist theorem requires sampling at >2x maximum frequency of interest
- 50 Hz provides adequate margin while minimizing power/data burden
- Matches typical IMU sensor capabilities and firmware timing constraints

### Why 50-Sample Windows?
- Empirical determination that most gestures complete within ~1 second
- Provides sufficient temporal context for CNN to learn motion patterns
- Balances latency requirements with gesture completeness
- Enables fixed-size input for batch processing in ML pipeline

## Usage Notes for Reproduction

1. **Physical Connection**: Ensure MPU6050 VCC to 3.3V, GND to ground, SDA to GPIO21, SCL to GPIO22 on ESP32
2. **AD0 Connection**: Connect AD0 pin to GND to fix I²C address at 0x68 (alternative address 0x69 if connected to 3.3V)
3. **Level Shifting**: Not required as both ESP32 and MPU6050 operate at 3.3V logic
4. **Pull-up Resistors**: Internal ESP32 pull-ups typically sufficient for short breadboard runs; external 4.7kΩ resistors recommended for longer traces or noisy environments
5. **Sensor Orientation**: Consistent mounting orientation important for replicating training data geometry