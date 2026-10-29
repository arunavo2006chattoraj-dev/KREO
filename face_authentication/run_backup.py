
import sys
import subprocess
import importlib.util
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

AUTH_SCRIPT = PROJECT_DIR / "authentication" / "authenticate_user.py"
ARDUINO_FILE = PROJECT_DIR / "arduino_controller.py"

BASE_ANGLE = 50
SHOULDER_ANGLE = 60


def authenticate():
    print("\n=== FACE AUTHENTICATION ===")

    if not AUTH_SCRIPT.exists():
        print(f"Authentication file not found: {AUTH_SCRIPT}")
        return False

    result = subprocess.run(
        [sys.executable, str(AUTH_SCRIPT)],
        cwd=str(PROJECT_DIR)
    )

    if result.returncode != 0:
        print("Authentication process failed.")
        return False

    print("Authentication process completed.")
    return True


def start_arduino():
    print("\n=== STARTING ARDUINO CONTROLLER ===")

    if not ARDUINO_FILE.exists():
        print(f"Arduino file not found: {ARDUINO_FILE}")
        return

    spec = importlib.util.spec_from_file_location(
        "arduino_controller", ARDUINO_FILE
    )

    controller = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = controller
    spec.loader.exec_module(controller)

    if not hasattr(controller, "connect"):
        print("connect() was not found in arduino_controller.py")
        return

    controller.connect()

    print("Arduino connection initialized.")

    if not hasattr(controller, "send_angles"):
        print("send_angles() was not found.")
        return

    controller.send_angles(BASE_ANGLE, SHOULDER_ANGLE)

    print(f"Base: {BASE_ANGLE} degrees")
    print(f"Shoulder: {SHOULDER_ANGLE} degrees")

    print("\n=== KREO BACKUP CONTROL ===")
    print("P - Pick object")
    print("D - Drop object")
    print("Q - Quit")

    try:
        while True:
            command = input("\nEnter command: ").strip().lower()

            if command == "p":
                if hasattr(controller, "pick_object"):
                    controller.pick_object()
                else:
                    print("pick_object() was not found.")

            elif command == "d":
                if hasattr(controller, "drop_object"):
                    controller.drop_object()
                else:
                    print("drop_object() was not found.")

            elif command == "q":
                print("Exiting backup control.")
                break

            else:
                print("Invalid command. Use P, D, or Q.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        if hasattr(controller, "disconnect"):
            controller.disconnect()


if __name__ == "__main__":
    print("========== KREO BACKUP MODE ==========")

    if authenticate():
        start_arduino()
    else:
        print("Authentication unsuccessful. Arduino will not start.")
