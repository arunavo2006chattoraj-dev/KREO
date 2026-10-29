
import cv2
import numpy as np
import threading

# ================= CONFIGURATION =================
CAMERA_ID = 0

ROWS = 5
COLS = 10

CROP_TOP = 45
CROP_BOTTOM = 78
CROP_LEFT = 2
CROP_RIGHT = 55

MIN_OBJECT_AREA = 100
LINE_MERGE_GAP = 12

# Shared object position
latest_position = None
position_lock = threading.Lock()

# =================================================


def get_position():
    """Return the latest detected (x, y) grid coordinates."""
    with position_lock:
        return latest_position


def set_position(position):
    """Safely update the shared grid coordinates."""
    global latest_position
    with position_lock:
        latest_position = position


def get_line_positions(line_image, axis, merge_gap=12):
    projection = np.sum(line_image > 0, axis=axis)

    if projection.size == 0 or projection.max() == 0:
        return []

    threshold = max(5, int(projection.max() * 0.25))
    positions = np.where(projection >= threshold)[0]

    if len(positions) == 0:
        return []

    groups = []
    start = previous = positions[0]

    for p in positions[1:]:
        if p - previous > merge_gap:
            groups.append((start, previous))
            start = p
        previous = p

    groups.append((start, previous))

    return [int((a + b) / 2) for a, b in groups]


def find_cell(x, y, xs, ys):
    """
    Find the cell containing the object.
    Return (row, column), using 1-based coordinates.
    """
    if len(xs) < 2 or len(ys) < 2:
        return None, None

    xs = sorted(xs)
    ys = sorted(ys)

    # Use the outermost detected grid boundaries.
    left, right = xs[0], xs[-1]
    top, bottom = ys[0], ys[-1]

    if right <= left or bottom <= top:
        return None, None

    if not (left <= x <= right and top <= y <= bottom):
        return None, None

    # Divide the boundary into 10 columns and 5 rows.
    cell_width = (right - left) / COLS
    cell_height = (bottom - top) / ROWS

    col = int((x - left) / cell_width) + 1
    row = int((y - top) / cell_height) + 1

    col = min(col, COLS)
    row = min(row, ROWS)

    return row, col


def run_grid():
    """Run camera and grid detection."""

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Camera not found. Try CAMERA_ID = 0 or 1.")
        set_position(None)
        return

    try:
        while True:

            # ================ READ CAMERA =================
            ret, frame = cap.read()

            if not ret:
                print("Camera frame unavailable.")
                set_position(None)
                break

            height, width = frame.shape[:2]

            # ================ CROP REGION =================
            y1 = int(height * CROP_TOP / 100)
            y2 = int(height * CROP_BOTTOM / 100)
            x1 = int(width * CROP_LEFT / 100)
            x2 = int(width * CROP_RIGHT / 100)

            roi = frame[y1:y2, x1:x2].copy()
            h, w = roi.shape[:2]

            # Green crop boundary
            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                (0, 255, 0), 3
            )

            cv2.putText(
                frame,
                "Grid Detection Region",
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

            # ================ GRID DETECTION ===============
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            binary = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                31, 7
            )

            horizontal_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (max(20, w // 15), 1)
            )

            vertical_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (1, max(20, h // 12))
            )

            horizontal = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                horizontal_kernel
            )

            vertical = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                vertical_kernel
            )

            ys = get_line_positions(
                horizontal, axis=1,
                merge_gap=LINE_MERGE_GAP
            )

            xs = get_line_positions(
                vertical, axis=0,
                merge_gap=LINE_MERGE_GAP
            )

            # ================ RED OBJECT DETECTION =========
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            lower_red1 = np.array([0, 80, 50])
            upper_red1 = np.array([15, 255, 255])

            lower_red2 = np.array([160, 80, 50])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

            red_mask = cv2.bitwise_or(mask1, mask2)

            kernel = np.ones((5, 5), np.uint8)

            red_mask = cv2.morphologyEx(
                red_mask, cv2.MORPH_OPEN, kernel
            )

            red_mask = cv2.morphologyEx(
                red_mask, cv2.MORPH_CLOSE, kernel
            )

            contours, _ = cv2.findContours(
                red_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # ================ GRID DISPLAY =================
            display = roi.copy()

            for yy in ys:
                cv2.line(
                    display, (0, yy), (w - 1, yy),
                    (255, 0, 0), 2
                )

            for xx in xs:
                cv2.line(
                    display, (xx, 0), (xx, h - 1),
                    (255, 0, 0), 2
                )

            # ================ OBJECT POSITION ==============
            valid_contours = [
                c for c in contours
                if cv2.contourArea(c) >= MIN_OBJECT_AREA
            ]

            message = "No red object detected"
            position = None

            if valid_contours:
                contour = max(
                    valid_contours,
                    key=cv2.contourArea
                )

                cv2.drawContours(
                    display, [contour], -1,
                    (0, 255, 0), 2
                )

                moments = cv2.moments(contour)

                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])

                    cv2.circle(
                        display, (cx, cy), 6,
                        (255, 0, 255), -1
                    )

                    row, col = find_cell(cx, cy, xs, ys)

                    if row is not None and col is not None:
                        # x = column; y = row
                        x = col
                        y = row

                        position = (x, y)
                        message = f"X: {x}, Y: {y}"
                    else:
                        message = "Object outside grid"

            # Share the latest position with the IK program.
            set_position(position)

            # ================ DISPLAY INFORMATION ==========
            cv2.putText(
                display, message, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

            cv2.putText(
                display,
                f"Horizontal lines: {len(ys)} (target {ROWS + 1})",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 2
            )

            cv2.putText(
                display,
                f"Vertical lines: {len(xs)} (target {COLS + 1})",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 2
            )

            # ================ DISPLAY WINDOWS ===============
            cv2.imshow("Camera - Green Crop Boundary", frame)
            cv2.imshow("Grid and Object Detection", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        set_position(None)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_grid()
