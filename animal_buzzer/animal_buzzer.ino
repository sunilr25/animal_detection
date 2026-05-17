const int buzzerPin = 2;
const int pirPin    = 4;  // GPIO 13 (D7 on NodeMCU) — PIR sensor output

// Send motion signal at most once every 500 ms to avoid flooding serial
unsigned long lastMotionSend = 0;
const unsigned long MOTION_SEND_INTERVAL = 500;

void setup() {
  Serial.begin(9600);
  pinMode(buzzerPin, OUTPUT);
  pinMode(pirPin, INPUT);
  noTone(buzzerPin);
}

void loop() {
  // ── PIR: relay motion to Python ──────────────────────────────────────────
  if (digitalRead(pirPin) == HIGH) {
    unsigned long now = millis();
    if (now - lastMotionSend >= MOTION_SEND_INTERVAL) {
      Serial.write('M');   // Python reads this to wake the camera
      lastMotionSend = now;
    }
  }

  // ── Buzzer: handle commands from Python ──────────────────────────────────
  if (Serial.available() > 0) {
    char command = Serial.read();

    if (command == 'R') {
      // Red alert: strong longer beep
      tone(buzzerPin, 1000);
      delay(250);
      noTone(buzzerPin);
    } else if (command == 'Y') {
      // Yellow alert: shorter beep
      tone(buzzerPin, 800);
      delay(120);
      noTone(buzzerPin);
    } else if (command == 'S') {
      // Stop any buzz
      noTone(buzzerPin);
    }
  }
}
