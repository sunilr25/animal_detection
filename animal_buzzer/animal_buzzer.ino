const int buzzerPin = 15;

void setup() {
  Serial.begin(9600);
  pinMode(buzzerPin, OUTPUT);
  noTone(buzzerPin);
}

void loop() {
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
