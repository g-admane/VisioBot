const int gripPin = 2; // Pin connected to the Grip relay
const int releasePin = 3; // Pin connected to the Release relay
const int upPin = 4; // Pin connected to the Up relay
const int downPin = 5; // Pin connected to the Down relay
const int leftPin = 6; // Pin connected to the Left relay
const int rightPin = 7; // Pin connected to the Right relay

unsigned long lastCommandTime = 0; // Stores the time of the last received command

void setup() {
  Serial.begin(9600); // Start serial communication
  pinMode(gripPin, OUTPUT);
  pinMode(releasePin, OUTPUT);
  pinMode(upPin, OUTPUT);
  pinMode(downPin, OUTPUT);
  pinMode(leftPin, OUTPUT);
  pinMode(rightPin, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
    lastCommandTime = millis(); // Update last command time on receiving new data
    String command = Serial.readStringUntil('\n'); // Read the incoming command
    Serial.println(command); // Print the command to the serial monitor

    // Actuate the relays based on the command  G11000
    digitalWrite(gripPin, command.charAt(1) == '1' ? HIGH : LOW);
    digitalWrite(releasePin, command.charAt(1) == '0' ? HIGH : LOW);
    digitalWrite(upPin, command.charAt(2) == '1' ? LOW : HIGH);
    digitalWrite(downPin, command.charAt(3) == '1' ? LOW : HIGH);
    digitalWrite(leftPin, command.charAt(4) == '1' ? LOW : HIGH);
    digitalWrite(rightPin, command.charAt(5) == '1' ? LOW : HIGH);
  } else {
    // Check if 500ms have passed since the last command
    if (millis() - lastCommandTime >= 500) {
      digitalWrite(upPin, HIGH);
      digitalWrite(downPin, HIGH);
      digitalWrite(leftPin, HIGH);
      digitalWrite(rightPin, HIGH);
    }
  }
}