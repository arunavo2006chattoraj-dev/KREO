
#include <Servo.h>

Servo Servo1;  // Base
Servo Servo2;  // Elbow - disabled
Servo Servo3;  // Shoulder
Servo Servo4;  // Gripper

const int SERVO1_PIN = 11;
const int SERVO2_PIN = 9;
const int SERVO3_PIN = 6;
const int SERVO4_PIN = 3;

const int STEP_DELAY = 50;

// Home positions
const int HOME_BASE = 0;
const int HOME_ELBOW = 30;
const int HOME_SHOULDER = 10;
const int HOME_GRIPPER = 0;

// Delivery position
const int DROP_BASE = 120;

// Gripper positions
const int GRIPPER_CLOSED = 100;
const int GRIPPER_OPEN = HOME_GRIPPER;

// Current positions
int basePos = HOME_BASE;
int shoulderPos = HOME_SHOULDER;
int gripperPos = HOME_GRIPPER;

// Manually entered target angles
int targetBase = 69;
int targetShoulder = 70;

// Robot state
bool anglesConfigured = false;
bool holdingObject = false;
bool busy = false;

String input = "";

void moveServo(Servo &servo, int &currentPos,
               int targetPos, const char* servoName) {

  targetPos = constrain(targetPos, 0, 180);

  Serial.print("[START] ");
  Serial.print(servoName);
  Serial.print(": ");
  Serial.print(currentPos);
  Serial.print(" -> ");
  Serial.println(targetPos);

  while (currentPos != targetPos) {
    if (currentPos < targetPos) {
      currentPos++;
    } else {
      currentPos--;
    }

    servo.write(currentPos);
    delay(STEP_DELAY);
  }

  Serial.print("[FINISHED] ");
  Serial.println(servoName);
}

void pick() {
  if (busy) return;

  if (!anglesConfigured) {
    Serial.println("ERROR_ANGLES_NOT_SET");
    return;
  }

  if (holdingObject) {
    Serial.println("ALREADY_HOLDING");
    return;
  }

  busy = true;
  Serial.println("PICKUP_STARTED");

  moveServo(Servo1, basePos, targetBase, "Base servo");

  moveServo(Servo3, shoulderPos, targetShoulder,
            "Shoulder servo");

  moveServo(Servo4, gripperPos, GRIPPER_CLOSED,
            "Gripper closing");

  delay(500);

  holdingObject = true;
  busy = false;

  Serial.println("PICKUP_SUCCESS");
}

void place() {
  if (busy) return;

  if (!holdingObject) {
    Serial.println("NOT_HOLDING_OBJECT");
    return;
  }

  busy = true;
  Serial.println("DELIVERY_STARTED");

  moveServo(Servo1, basePos, DROP_BASE,
            "Base to drop position");

  moveServo(Servo4, gripperPos, GRIPPER_OPEN,
            "Gripper opening");

  delay(500);

  // The object has been released according to the
  // programmed sequence. No physical sensor is used.
  holdingObject = false;

  Serial.println("DELIVERY_SUCCESS");

  moveServo(Servo3, shoulderPos, HOME_SHOULDER,
            "Shoulder home");

  moveServo(Servo1, basePos, HOME_BASE,
            "Base home");

  busy = false;

  Serial.println("RETURN_HOME_COMPLETE");
}

void returnHome() {
  if (busy) return;

  if (holdingObject) {
    Serial.println("HOME_DENIED_OBJECT_HELD");
    return;
  }

  busy = true;

  moveServo(Servo4, gripperPos, GRIPPER_OPEN,
            "Gripper home");

  moveServo(Servo3, shoulderPos, HOME_SHOULDER,
            "Shoulder home");

  moveServo(Servo1, basePos, HOME_BASE,
            "Base home");

  busy = false;

  Serial.println("HOME_REACHED");
}

void processCommand(String data) {
  data.trim();

  String command = data;
  command.toUpperCase();

  if (command == "PICK") {
    pick();
    return;
  }

  if (command == "DROP") {
    place();
    return;
  }

  if (command == "HOME") {
    returnHome();
    return;
  }

  if (command == "STATUS") {
    Serial.println(holdingObject
                   ? "STATUS_HOLDING"
                   : "STATUS_EMPTY");
    return;
  }

  // Otherwise, expect manually entered base,shoulder angles
  int comma = data.indexOf(',');

  if (comma <= 0 || comma == data.length() - 1) {
    Serial.println(
      "ERROR_USE_ANGLE_FORMAT_BASE,SHOULDER_OR_PICK_DROP_HOME"
    );
    return;
  }

  String first = data.substring(0, comma);
  String second = data.substring(comma + 1);

  first.trim();
  second.trim();

  // Ensure both values contain only digits.
  for (unsigned int i = 0; i < first.length(); i++) {
    if (!isDigit(first[i])) {
      Serial.println("ERROR_INVALID_ANGLE");
      return;
    }
  }

  for (unsigned int i = 0; i < second.length(); i++) {
    if (!isDigit(second[i])) {
      Serial.println("ERROR_INVALID_ANGLE");
      return;
    }
  }

  int theta1 = first.toInt();
  int theta2 = second.toInt();

  if (theta1 < 0 || theta1 > 180 ||
      theta2 < 0 || theta2 > 180) {
    Serial.println("ERROR_ANGLES_OUT_OF_RANGE");
    return;
  }

  if (holdingObject || busy) {
    Serial.println("ERROR_CANNOT_CHANGE_ANGLES");
    return;
  }

  targetBase = theta1;
  targetShoulder = theta2;
  anglesConfigured = true;

  Serial.print("ANGLES_SET:");
  Serial.print(targetBase);
  Serial.print(",");
  Serial.println(targetShoulder);
}

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(1000);

  Serial.println("Initializing servos...");

  Servo1.attach(SERVO1_PIN);
  // Servo2.attach(SERVO2_PIN); // Elbow intentionally disabled
  Servo3.attach(SERVO3_PIN);
  Servo4.attach(SERVO4_PIN);

  Servo1.write(basePos);
  Servo3.write(shoulderPos);
  Servo4.write(gripperPos);

  delay(1000);

  Serial.println("ROBOT_READY");
  Serial.println("Enter base,shoulder angles, e.g. 69,70");
  Serial.println("Then send PICK or DROP");
}

void loop() {
  while (Serial.available() > 0) {
    char ch = Serial.read();

    if (ch == '\n' || ch == '\r') {
      if (input.length() > 0) {
        String command = input;
        input = "";
        processCommand(command);
      }
    } else if (input.length() < 64) {
      input += ch;
    }
  }
}
