
import serial
import time
import threading

ARDUINO_PORT = "COM5"
BAUD_RATE = 9600

arduino = None
serial_lock = threading.Lock()


def connect():
    """Connect to Arduino and wait for it to reset."""
    global arduino

    try:
        arduino = serial.Serial(
            ARDUINO_PORT,
            BAUD_RATE,
            timeout=1
        )

        time.sleep(2)
        arduino.reset_input_buffer()

        print(f"Arduino connected on {ARDUINO_PORT}.")
        return True

    except (serial.SerialException, OSError) as error:
        print(f"Arduino connection failed: {error}")
        arduino = None
        return False


def read_response(expected_responses, timeout=120):
    """Read Arduino messages until the expected response arrives."""
    if arduino is None or not arduino.is_open:
        print("Arduino is not connected.")
        return None

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        try:
            response = arduino.readline().decode(
                "utf-8",
                errors="replace"
            ).strip()
        except (serial.SerialException, OSError) as error:
            print(f"Arduino read error: {error}")
            return None

        if not response:
            continue

        print("Arduino:", response)

        if response in expected_responses:
            return response

        # Stop on error or a known rejected command.
        if response.startswith("ERROR"):
            return response

        if response in {
            "ALREADY_HOLDING",
            "NOT_HOLDING_OBJECT",
            "HOME_DENIED_OBJECT_HELD"
        }:
            return response

    print("Timed out waiting for Arduino response.")
    return None


def send_command(command, expected_responses, timeout=120):
    """Send one command and wait for its expected response."""
    if arduino is None or not arduino.is_open:
        print("Arduino is not connected.")
        return False

    try:
        with serial_lock:
            arduino.write((command + "\n").encode("utf-8"))
            arduino.flush()

            print("Sent to Arduino:", command)

            response = read_response(
                expected_responses,
                timeout
            )

            return response in expected_responses

    except (serial.SerialException, OSError) as error:
        print(f"Arduino communication error: {error}")
        return False


def send_angles(theta1, theta2):
    """Configure base and shoulder target angles."""
    try:
        theta1 = float(theta1)
        theta2 = float(theta2)

        if not (
            theta1.is_integer()
            and theta2.is_integer()
            and 0 <= theta1 <= 180
            and 0 <= theta2 <= 180
        ):
            print("Angles must be whole numbers from 0 to 180.")
            return False

        base = int(theta1)
        shoulder = int(theta2)

        return send_command(
            f"{base},{shoulder}",
            {f"ANGLES_SET:{base},{shoulder}"},
            timeout=10
        )

    except (ValueError, TypeError):
        print("Invalid angles.")
        return False


def pick_object():
    """Pick up the object at the configured angles."""
    return send_command(
        "PICK",
        {"PICKUP_SUCCESS"},
        timeout=120
    )


def drop_object():
    """
    Release the object and wait until the programmed
    return-home sequence has completed.
    """
    return send_command(
        "DROP",
        {"RETURN_HOME_COMPLETE"},
        timeout=120
    )


def return_home():
    """Return home when the robot is not holding an object."""
    return send_command(
        "HOME",
        {"HOME_REACHED"},
        timeout=120
    )


def disconnect():
    """Close the Arduino serial connection."""
    global arduino

    if arduino is not None:
        try:
            if arduino.is_open:
                arduino.close()
                print("Arduino disconnected.")

        except (serial.SerialException, OSError) as error:
            print("Disconnect error:", error)

        finally:
            arduino = None
