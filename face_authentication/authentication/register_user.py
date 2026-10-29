import cv2
import numpy as np
import os
import time
import onnxruntime as ort
from insightface.app import FaceAnalysis


# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 0

NUMBER_OF_SAMPLES = 20

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

DETECTION_SIZE = (640, 640)

CAPTURE_INTERVAL = 0.20


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "face_database"
)

MODEL_ROOT = os.path.join(
    BASE_DIR,
    "insightface_models"
)

os.makedirs(DATABASE_PATH, exist_ok=True)
os.makedirs(MODEL_ROOT, exist_ok=True)


# ============================================================
# GPU / ONNX RUNTIME CHECK
# ============================================================

print("=" * 60)
print("GPU / ONNX RUNTIME CHECK")
print("=" * 60)

providers = ort.get_available_providers()

print("Available providers:")
print(providers)

if "CUDAExecutionProvider" in providers:

    print("\nCUDAExecutionProvider detected.")
    print("RTX GPU will be used.")

    INSIGHTFACE_PROVIDERS = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]

else:

    print("\nWARNING:")
    print("CUDAExecutionProvider is NOT available.")
    print("Face registration will run on CPU.")

    INSIGHTFACE_PROVIDERS = [
        "CPUExecutionProvider"
    ]


# ============================================================
# USER NAME
# ============================================================

print("\n" + "=" * 60)

USER_NAME = input(
    "Enter user name: "
).strip()

if not USER_NAME:

    print("\nERROR: User name cannot be empty.")
    exit()


# ============================================================
# USER DIRECTORY
# ============================================================

USER_PATH = os.path.join(
    DATABASE_PATH,
    USER_NAME
)

os.makedirs(
    USER_PATH,
    exist_ok=True
)

print("\nUser:")
print(USER_NAME)

print("\nSaving embeddings to:")
print(USER_PATH)


# ============================================================
# MODEL DIRECTORY
# ============================================================

print("\n" + "=" * 60)
print("INSIGHTFACE MODEL DIRECTORY")
print("=" * 60)

print(MODEL_ROOT)


# ============================================================
# INITIALIZE INSIGHTFACE
# ============================================================

print("\n" + "=" * 60)
print("LOADING INSIGHTFACE")
print("=" * 60)

try:

    app = FaceAnalysis(
        name="buffalo_l",
        root=MODEL_ROOT,
        providers=INSIGHTFACE_PROVIDERS
    )

    app.prepare(
        ctx_id=0,
        det_size=DETECTION_SIZE
    )

except Exception as e:

    print("\nERROR: Could not load InsightFace.")
    print("Details:")
    print(e)

    print("\nModel directory:")
    print(MODEL_ROOT)

    exit()


print("\nInsightFace loaded successfully.")


# ============================================================
# OPEN CAMERA
# ============================================================

print("\n" + "=" * 60)
print("OPENING CAMERA")
print("=" * 60)

cap = cv2.VideoCapture(
    CAMERA_INDEX
)

if not cap.isOpened():

    print("\nERROR: Could not open webcam.")

    exit()


# ============================================================
# CAMERA SETTINGS
# ============================================================

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

cap.set(
    cv2.CAP_PROP_FPS,
    CAMERA_FPS
)


actual_width = cap.get(
    cv2.CAP_PROP_FRAME_WIDTH
)

actual_height = cap.get(
    cv2.CAP_PROP_FRAME_HEIGHT
)

actual_fps = cap.get(
    cv2.CAP_PROP_FPS
)


print("Camera width :", actual_width)
print("Camera height:", actual_height)
print("Camera FPS   :", actual_fps)


# ============================================================
# REGISTRATION VARIABLES
# ============================================================

embeddings = []

previous_time = time.perf_counter()

fps = 0.0

last_capture_time = 0.0


# ============================================================
# INSTRUCTIONS
# ============================================================

print("\n" + "=" * 60)
print("FACE REGISTRATION")
print("=" * 60)

print("Look directly at the camera.")
print("Keep only ONE person in the frame.")
print("Move your head slightly while registering.")
print("Do not move too quickly.")
print("Press Q to cancel.")

print(
    f"\nRequired samples: {NUMBER_OF_SAMPLES}"
)

print("\nStarting camera...")


# ============================================================
# REGISTRATION LOOP
# ============================================================

while len(embeddings) < NUMBER_OF_SAMPLES:

    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:

        print("\nERROR: Could not read camera frame.")

        break


    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    ai_start = time.perf_counter()

    faces = app.get(frame)

    ai_end = time.perf_counter()

    ai_time = ai_end - ai_start


    # --------------------------------------------------------
    # FPS CALCULATION
    # --------------------------------------------------------

    current_time = time.perf_counter()

    elapsed = (
        current_time -
        previous_time
    )

    if elapsed > 0:

        current_fps = 1.0 / elapsed

        if fps == 0:

            fps = current_fps

        else:

            fps = (
                0.9 * fps +
                0.1 * current_fps
            )

    previous_time = current_time


    # --------------------------------------------------------
    # DISPLAY FRAME
    # --------------------------------------------------------

    display_frame = frame.copy()


    # --------------------------------------------------------
    # NO FACE
    # --------------------------------------------------------

    if len(faces) == 0:

        cv2.putText(
            display_frame,
            "NO FACE DETECTED",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # --------------------------------------------------------
    # MULTIPLE FACES
    # --------------------------------------------------------

    elif len(faces) > 1:

        cv2.putText(
            display_frame,
            "ONLY ONE PERSON ALLOWED",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # --------------------------------------------------------
    # ONE FACE
    # --------------------------------------------------------

    else:

        face = faces[0]


        # ----------------------------------------------------
        # FACE BOX
        # ----------------------------------------------------

        x1, y1, x2, y2 = (
            face.bbox.astype(int)
        )

        cv2.rectangle(
            display_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # ----------------------------------------------------
        # CAPTURE EMBEDDING
        # ----------------------------------------------------

        now = time.perf_counter()

        if (
            now - last_capture_time
            >= CAPTURE_INTERVAL
        ):

            embedding = face.embedding.copy()


            # ----------------------------------------------
            # NORMALIZE EMBEDDING
            # ----------------------------------------------

            norm = np.linalg.norm(
                embedding
            )

            if norm > 0:

                embedding = (
                    embedding / norm
                )

                embeddings.append(
                    embedding
                )

                last_capture_time = now

                print(
                    f"Captured sample "
                    f"{len(embeddings)}/"
                    f"{NUMBER_OF_SAMPLES}"
                )


        # ----------------------------------------------------
        # SAMPLE COUNT
        # ----------------------------------------------------

        cv2.putText(
            display_frame,
            f"Samples: "
            f"{len(embeddings)}/"
            f"{NUMBER_OF_SAMPLES}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    # ========================================================
    # PERFORMANCE DISPLAY
    # ========================================================

    cv2.putText(
        display_frame,
        f"FPS: {fps:.1f}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )


    cv2.putText(
        display_frame,
        f"AI: {ai_time * 1000:.0f} ms",
        (30, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "GPU Face Registration",
        display_frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        print(
            "\nRegistration cancelled."
        )

        break


# ============================================================
# RELEASE CAMERA
# ============================================================

cap.release()

cv2.destroyAllWindows()


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

if len(embeddings) == NUMBER_OF_SAMPLES:

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # SAVE PATH
    # --------------------------------------------------------

    save_path = os.path.join(
        USER_PATH,
        "embeddings.npy"
    )


    np.save(
        save_path,
        embeddings
    )


    # ========================================================
    # SUCCESS
    # ========================================================

    print("\n" + "=" * 60)
    print("REGISTRATION SUCCESSFUL")
    print("=" * 60)

    print("User:")
    print(USER_NAME)

    print(
        "Samples:",
        len(embeddings)
    )

    print(
        "Embedding shape:",
        embeddings.shape
    )

    print("\nSaved to:")
    print(save_path)

    print("\nModel files are stored in:")
    print(MODEL_ROOT)


else:

    # ========================================================
    # NOT COMPLETED
    # ========================================================

    print("\n" + "=" * 60)
    print("REGISTRATION NOT COMPLETED")
    print("=" * 60)

    print(
        f"Captured {len(embeddings)} "
        f"of {NUMBER_OF_SAMPLES} samples."
    )

    print("\nNo embedding file was saved.")