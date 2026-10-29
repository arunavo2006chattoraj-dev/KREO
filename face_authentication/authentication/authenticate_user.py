
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis



USER_NAME = "r"
EMBEDDING_FILENAME = "embeddings.npy"

THRESHOLD = 0.35

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

MODEL_NAME = "buffalo_l"
DETECTION_SIZE = (640, 640)
MODEL_FOLDER = "insightface_models"


# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "face_database"
)

USER_PATH = os.path.join(
    DATABASE_PATH,
    USER_NAME
)

EMBEDDING_PATH = os.path.join(
    USER_PATH,
    EMBEDDING_FILENAME
)

MODEL_ROOT = os.path.join(
    BASE_DIR,
    MODEL_FOLDER
)


# ==========================================
# ERROR HANDLING
# ==========================================

def stop_program(message):
    print("\n")
    print(message)
    print("\n")
    raise SystemExit(1)


# ==========================================
# GPU / ONNX RUNTIME CHECK
# ==========================================

print("\nGPU / ONNX RUNTIME CHECK")
print("-" * 30)

providers = ort.get_available_providers()

print("\nAvailable providers:")

for provider in providers:
    print("  -", provider)

if "CUDAExecutionProvider" in providers:
    print("\nCUDAExecutionProvider detected.")
    print("InsightFace may use the NVIDIA GPU.")
else:
    print("\nWARNING: CUDAExecutionProvider is not available.")
    print("InsightFace may run on the CPU.")


# ==========================================
# CHECK FACE DATABASE
# ==========================================

if not os.path.isdir(DATABASE_PATH):
    stop_program(
        "ERROR: FACE DATABASE NOT FOUND\n\n"
        f"Expected database:\n{DATABASE_PATH}"
    )


# ==========================================
# CHECK SELECTED USER
# ==========================================

if not os.path.isdir(USER_PATH):
    print("\nERROR: USER DATASET NOT FOUND")
    print("\nSelected user:")
    print(USER_NAME)
    print("\nExpected folder:")
    print(USER_PATH)

    print("\nAvailable datasets:")

    available_users = [
        item
        for item in os.listdir(DATABASE_PATH)
        if os.path.isdir(os.path.join(DATABASE_PATH, item))
    ]

    if available_users:
        for user in sorted(available_users):
            print("  -", user)
    else:
        print("  No user datasets found.")

    raise SystemExit(1)


if not os.path.isfile(EMBEDDING_PATH):
    stop_program(
        "ERROR: EMBEDDING FILE NOT FOUND\n\n"
        f"Expected file:\n{EMBEDDING_PATH}"
    )


# ==========================================
# LOAD REGISTERED EMBEDDINGS
# ==========================================

print("\nLOADING REGISTERED EMBEDDINGS")
print("-" * 30)

try:
    stored_embeddings = np.load(
        EMBEDDING_PATH,
        allow_pickle=False
    )
except Exception as error:
    stop_program(
        "ERROR: COULD NOT LOAD EMBEDDING FILE\n\n"
        f"Details:\n{error}"
    )

try:
    stored_embeddings = np.asarray(
        stored_embeddings,
        dtype=np.float32
    )
except Exception as error:
    stop_program(
        "ERROR: EMBEDDING DATA IS NOT NUMERIC\n\n"
        f"Details:\n{error}"
    )


# Handle a single embedding
if stored_embeddings.ndim == 1:
    stored_embeddings = stored_embeddings.reshape(1, -1)


# Validate embedding dimensions
if stored_embeddings.ndim != 2:
    stop_program(
        "ERROR: INVALID EMBEDDING SHAPE\n\n"
        f"Received shape: {stored_embeddings.shape}\n"
        "Expected: (number_of_samples, embedding_dimension)"
    )

if stored_embeddings.shape[0] == 0:
    stop_program("ERROR: EMBEDDING DATABASE IS EMPTY")

if stored_embeddings.shape[1] != 512:
    stop_program(
        "ERROR: INVALID EMBEDDING DIMENSION\n\n"
        f"Received dimension: {stored_embeddings.shape[1]}\n"
        "Expected dimension: 512"
    )

print("\nRegistered embeddings loaded.")
print("Embedding shape:", stored_embeddings.shape)
print("Number of registered samples:", stored_embeddings.shape[0])


# ==========================================
# NORMALIZE REGISTERED EMBEDDINGS
# ==========================================

print("\nNormalizing registered embeddings...")

stored_norms = np.linalg.norm(
    stored_embeddings,
    axis=1,
    keepdims=True
)

zero_norm_mask = stored_norms[:, 0] == 0

if np.any(zero_norm_mask):
    invalid_count = int(np.sum(zero_norm_mask))

    print(
        f"WARNING: {invalid_count} zero-length embedding(s) found."
    )

    stored_norms[zero_norm_mask] = 1.0

stored_embeddings = stored_embeddings / stored_norms

stored_embeddings = np.ascontiguousarray(
    stored_embeddings,
    dtype=np.float32
)

print("Embedding normalization complete.")


# ==========================================
# LOAD INSIGHTFACE
# ==========================================

print("\nLOADING INSIGHTFACE\n")

try:
    app = FaceAnalysis(
        name=MODEL_NAME,
        root=MODEL_ROOT
    )

    app.prepare(
        ctx_id=0,
        det_size=DETECTION_SIZE
    )

except Exception as error:
    stop_program(
        "ERROR: COULD NOT LOAD INSIGHTFACE\n\n"
        f"Details:\n{error}\n\n"
        "Expected model location:\n"
        f"{os.path.join(MODEL_ROOT, 'models', MODEL_NAME)}"
    )

print("InsightFace loaded successfully.")


# ==========================================
# OPEN CAMERA
# ==========================================

print("\nOPENING CAMERA")

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    cap.release()
    stop_program("ERROR: COULD NOT OPEN WEBCAM")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_camera_fps = cap.get(cv2.CAP_PROP_FPS)

print("Camera width :", actual_width)
print("Camera height:", actual_height)
print("Camera FPS   :", actual_camera_fps)


# ==========================================
# AUTHENTICATION STATE
# ==========================================

authenticated = False

previous_frame_time = time.perf_counter()
fps = 0.0
frame_count = 0


# ==========================================
# AUTHENTICATION LOOP
# ==========================================

print("\nCAMERA + AUTHENTICATION STARTED")
print("\nAuthenticating against dataset:", USER_NAME)
print("Authentication threshold:", THRESHOLD)
print("Look at the camera.")
print("Press Q to exit.")


try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("\nERROR: COULD NOT READ CAMERA FRAME.")
            break

        frame_count += 1

        # Face detection
        ai_start = time.perf_counter()
        faces = app.get(frame)
        ai_end = time.perf_counter()

        ai_time = ai_end - ai_start

        # FPS calculation
        current_time = time.perf_counter()
        frame_interval = current_time - previous_frame_time

        if frame_interval > 0:
            instant_fps = 1.0 / frame_interval

            if fps == 0.0:
                fps = instant_fps
            else:
                fps = 0.90 * fps + 0.10 * instant_fps

        previous_frame_time = current_time

        display_frame = frame.copy()
        similarity = 0.0

        # No face detected
        if len(faces) == 0:
            cv2.putText(
                display_frame,
                "NO FACE",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # More than one face detected
        elif len(faces) > 1:
            cv2.putText(
                display_frame,
                "ONLY ONE PERSON",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            for detected_face in faces:
                x1, y1, x2, y2 = (
                    detected_face.bbox.astype(int)
                )

                cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

        # Exactly one face detected
        else:
            face = faces[0]

            x1, y1, x2, y2 = face.bbox.astype(int)

            live_embedding = np.asarray(
                face.embedding,
                dtype=np.float32
            )

            live_norm = np.linalg.norm(live_embedding)

            if live_norm > 0:
                live_embedding = live_embedding / live_norm

                similarities = np.dot(
                    stored_embeddings,
                    live_embedding
                )

                similarity = float(np.max(similarities))

            # Authentication decision
            if similarity >= THRESHOLD:
                authenticated = True
                status = f"AUTHORIZED: {USER_NAME}"
                box_color = (0, 255, 0)

            else:
                status = "ACCESS DENIED"
                box_color = (0, 0, 255)

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )

            text_y = max(y1 - 10, 30)

            cv2.putText(
                display_frame,
                status,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                box_color,
                2
            )

            cv2.putText(
                display_frame,
                f"Similarity: {similarity:.3f}",
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_color,
                2
            )

        # Display FPS
        cv2.putText(
            display_frame,
            f"FPS: {fps:.1f}",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # Display AI processing time
        cv2.putText(
            display_frame,
            f"AI: {ai_time * 1000:.0f} ms",
            (30, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # Display dataset name
        cv2.putText(
            display_frame,
            f"Dataset: {USER_NAME}",
            (30, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # Display number of samples
        cv2.putText(
            display_frame,
            f"Samples: {stored_embeddings.shape[0]}",
            (30, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.imshow(
            "GPU Face Authentication",
            display_frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

except KeyboardInterrupt:
    print("\nAuthentication interrupted.")

except Exception as error:
    print("\nERROR during authentication:", error)
    authenticated = False

finally:
    cap.release()
    cv2.destroyAllWindows()


# ==========================================
# RETURN RESULT TO MAIN.PY
# ==========================================

if authenticated:
    print("\n[AUTH_RESULT] SUCCESS")
    raise SystemExit(0)

else:
    print("\n[AUTH_RESULT] FAILURE")
    raise SystemExit(1)
