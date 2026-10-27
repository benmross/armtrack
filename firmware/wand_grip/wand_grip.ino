// Wand grip: an MPU-6050 gyro, a gripper trigger and three buttons, streamed to
// the host at 200 Hz as fixed 22-byte packets (see stereo_hand/wand.py,
// parse_packets). Binary, because 200 lines of text a second nearly fills
// 115200 baud and leaves no margin; each packet carries a checksum so the host
// can resync after a dropped byte.
//
// Wiring (Uno or Nano):
//   GY-521 VCC->5V GND->GND SCL->A5 SDA->A4  (the GY-521 has its own regulator)
//   gripper: the kit joystick's VRy -> A0 (rate mode, the default on the host),
//            or a trigger potentiometer's wiper -> A0 (--grip-mode absolute)
//   home: the joystick's SW pin can replace the D3 button
//   clutch button  D2 -> GND   (hold to drive the arm; release lets go)
//   home button    D3 -> GND
//   fine button    D4 -> GND   (hold for fine, scaled-down motion)
// Buttons use the internal pull-ups, so no resistors are needed.
//
// Mount the GY-521 flat in the grip with its printed X arrow pointing right and
// Y pointing forward (toward the cube's front face), chip side up: then the
// gyro's axes are the wand's axes and the host needs no remapping.

#include <Wire.h>

const uint8_t MPU = 0x68;
const uint8_t PIN_CLUTCH = 2, PIN_HOME = 3, PIN_FINE = 4, PIN_TRIGGER = A0;
const unsigned long PERIOD_US = 5000;  // 200 Hz

void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000);  // drop to 100000 if the cable to the grip is longer than ~1 m
  writeReg(0x6B, 0x01);   // PWR_MGMT_1: wake, clock from the X gyro PLL (steadier than the internal RC)
  writeReg(0x1A, 0x03);   // CONFIG: 44 Hz low-pass, below the 100 Hz Nyquist of a 200 Hz stream
  writeReg(0x19, 0x04);   // SMPLRT_DIV: 1 kHz / 5 = 200 Hz
  writeReg(0x1B, 0x08);   // GYRO_CONFIG: +/-500 deg/s (65.5 LSB per deg/s)
  writeReg(0x1C, 0x08);   // ACCEL_CONFIG: +/-4 g (8192 LSB per g)
  pinMode(PIN_CLUTCH, INPUT_PULLUP);
  pinMode(PIN_HOME, INPUT_PULLUP);
  pinMode(PIN_FINE, INPUT_PULLUP);
}

void loop() {
  static unsigned long next = micros();
  while ((long)(micros() - next) < 0) {}
  next += PERIOD_US;
  unsigned long now = micros();

  int16_t a[3], g[3];
  Wire.beginTransmission(MPU);
  Wire.write(0x3B);  // ACCEL_XOUT_H; accel, temperature and gyro follow in one burst
  Wire.endTransmission(false);
  Wire.requestFrom(MPU, (uint8_t)14);
  for (int i = 0; i < 3; i++) a[i] = (Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read();  // temperature
  for (int i = 0; i < 3; i++) g[i] = (Wire.read() << 8) | Wire.read();

  uint16_t trig = analogRead(PIN_TRIGGER);
  uint8_t btn = (!digitalRead(PIN_CLUTCH)) | (!digitalRead(PIN_HOME) << 1) | (!digitalRead(PIN_FINE) << 2);

  uint8_t p[22];
  p[0] = 0xA5; p[1] = 0x5A;
  memcpy(p + 2, g, 6);   // AVR is little-endian, as the host's struct expects
  memcpy(p + 8, a, 6);
  memcpy(p + 14, &trig, 2);
  p[16] = btn;
  memcpy(p + 17, &now, 4);
  uint8_t sum = 0;
  for (int i = 2; i < 21; i++) sum += p[i];
  p[21] = sum;
  Serial.write(p, 22);
}
