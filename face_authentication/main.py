
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# Ensure the project root is available for imports.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import arduino_controller

AUTH_SCRIPT = PROJECT_DIR / "authentication" / "authenticate_user.py"

started = False
authenticated = False
object_picked = False


def startup():
    global started, authenticated

    started = True
    authenticated = False

    print("[MAIN] Robot voice control activated.")
    return True


def authenticate_face():
    global authenticated

    authenticated = False

    if not started:
        print("[MAIN] Say the wake command first.")
        return False

    if not AUTH_SCRIPT.is_file():
        print("[MAIN] Authentication file not found:")
        print(AUTH_SCRIPT)
        return False

    print("[MAIN] Starting face authentication...")

    try:
        result = subprocess.run(
            [sys.executable, str(AUTH_SCRIPT)],
            cwd=str(PROJECT_DIR),
            check=False
        )

        if result.returncode == 0:
            authenticated = True
            print("[MAIN] Authentication successful.")
            return True

        print("[MAIN] Authentication failed.")
        print("[MAIN] Exit code:", result.returncode)
        return False

    except Exception as error:
        print("[MAIN] Could not run authentication:", error)
        return False


def pick_object():
    global object_picked

    if not started:
        print("[MAIN] Robot is not active.")
        return False

    if not authenticated:
        print("[MAIN] Pickup denied: authenticate first.")
        return False

    if object_picked:
        print("[MAIN] An object is already marked as picked.")
        return False

    print("[MAIN] Sending PICK command to Arduino...")

    try:
        success = arduino_controller.pick_object()

        if success is True:
            object_picked = True
            print("[MAIN] Arduino reported pickup sequence complete.")
            return True

        print("[MAIN] Pickup failed or was not confirmed.")
        return False

    except Exception as error:
        print("[MAIN] Arduino pickup error:", error)
        return False


def drop_object():
    global object_picked

    if not started:
        print("[MAIN] Robot is not active.")
        return False

    if not authenticated:
        print("[MAIN] Drop denied: authenticate first.")
        return False

    if not object_picked:
        print("[MAIN] No object is marked as picked.")
        return False

    print("[MAIN] Sending DROP command to Arduino...")

    try:
        success = arduino_controller.drop_object()

        if success is True:
            object_picked = False
            print("[MAIN] Arduino reported delivery sequence complete.")
            return True

        print("[MAIN] Delivery failed or was not confirmed.")
        return False

    except Exception as error:
        print("[MAIN] Arduino drop error:", error)
        return False


def sleep_robot():
    global started, authenticated

    # Do not clear object_picked here:
    # the arm may still be holding an object.
    started = False
    authenticated = False

    print("[MAIN] Robot session stopped.")
    return True
