import sys
from pathlib import Path

# Add the project root and IK folder to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IK_FOLDER = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(IK_FOLDER))

import arduino_controller
import math
import threading
import time
import grid
# ================= CONFIGURATION =================
box_length = 4.5 * 4
l1 = 9.5
l2 = 14.5
g = 5.5



def inverse_kinematics(x, y):
    d = None
    theta1 = None
    theta2 = None

    if x < 5:
        d = math.sqrt(
            ((y ) * 4) ** 2
            + (box_length - x * 4) ** 2
        )

        theta1 = math.degrees(
            math.atan(
                ((y ) * 4) / (box_length - x * 4)
            )
        )

        value = (d + g) / (l1 + l2)

        if -1 <= value <= 1:
            theta2 = math.degrees(math.asin(value))
        else:
            print(
                "Theta2 cannot be calculated: "
                "asin input is outside [-1, 1]."
            )

    elif x == 5:
        theta1 = 95

        if y == 5:
            theta2 = 76
        elif y == 4:
            theta2 = 46

    else:
        d = math.sqrt(
            (x - 1) ** 2 + (y - box_length) ** 2
        )

        phi = math.degrees(
            math.atan(
                ((y ) * 4) / (-box_length + x * 4)
            )
        )

        theta1 = 180 - phi

   
    print(f"\nGrid position: x={x}, y={y}")

    if d is not None:
        print(f"Distance (d): {d:.2f}")

    if theta1 is not None:
        print(f"Theta1: {round(theta1)} degrees")

    if theta2 is not None:
        print(f"Theta2: {round(theta2)} degrees")

    if theta1 is None or theta2 is None:
        return None, None

    return round(theta1), round(theta2)

   



def main():
    if not arduino_controller.connect():
        print("Arduino connection failed. Check the COM port.")
        return

    grid_thread = threading.Thread(
        target=grid.run_grid,
        daemon=True
    )
    grid_thread.start()

    previous_position = None

    try:
        while grid_thread.is_alive():
            position = grid.get_position()

            if position is None:
                previous_position = None
            elif position != previous_position:
                # Record this position so it isn't sent repeatedly.
                previous_position = position

                x, y = position
                theta1, theta2 = inverse_kinematics(x, y)

                if theta1 is not None and theta2 is not None:
                    print(f"Sending angles: {theta1},{theta2}")

                    success = arduino_controller.send_angles(
                        theta1, theta2
                    )

                    if not success:
                        print("Arduino command failed.")
                else:
                    print("No valid angles for this grid position.")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Stopping inverse kinematics.")

    finally:
        arduino_controller.disconnect()
        print("Program stopped.")


if __name__ == "__main__":
    main()

    