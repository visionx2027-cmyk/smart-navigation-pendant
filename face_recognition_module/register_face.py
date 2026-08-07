import os
import cv2
import time
import pickle
import numpy as np
import face_recognition
import pyttsx3

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_PATH = os.path.join(KNOWN_FACES_DIR, "encodings.pkl")

# Poses to capture. Each is (label_for_display, voice_instruction).
POSES = [
    ("Look Straight", "Please look straight at the camera"),
    ("Turn Left", "Please turn your head slightly to the left"),
    ("Turn Right", "Please turn your head slightly to the right"),
    ("Tilt Up", "Please tilt your head slightly up"),
    ("Tilt Down", "Please tilt your head slightly down"),
]

FRAMES_TO_ACCEPT_PER_POSE = 5       # valid (post-filter) encodings per pose
MAX_ATTEMPTS_PER_POSE = 80           # safety cap so a pose can't loop forever
MIN_FACE_WIDTH_PX = 90               # reject faces smaller than this (too far)
BLUR_VARIANCE_THRESHOLD = 60.0       # below this = too blurry (Laplacian var)
ENCODING_MODEL = "large"             # 68-point landmarks, not "small"
REGISTRATION_JITTERS = 10            # higher quality, registration is offline
DETECTION_MODEL = "hog"              # "cnn" is more accurate but slow on CPU/Pi
UPSAMPLE_TIMES = 1                   # increase to 2 if faces are small/far

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# --------------------------------------------------------------------------
# Voice
# --------------------------------------------------------------------------
class Voice:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 165)

    def say(self, text: str, block: bool = True):
        print(f"[VOICE] {text}")
        self.engine.say(text)
        if block:
            self.engine.runAndWait()


# --------------------------------------------------------------------------
# Image quality helpers
# --------------------------------------------------------------------------
def sharpness_score(gray_face_crop: np.ndarray) -> float:
    """Variance of Laplacian — higher means sharper. Used to reject
    motion-blurred captures which produce unstable encodings."""
    return cv2.Laplacian(gray_face_crop, cv2.CV_64F).var()


def apply_clahe(bgr_frame: np.ndarray) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization on the L channel.
    Improves robustness to the uneven lighting that a turned face often
    produces (one side shadowed), without distorting color-based cues."""
    lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    merged = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------
def load_database() -> dict:
    """dict[name] -> list[np.ndarray] (each a 128-d encoding)."""
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_database(db: dict):
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(db, f)


# --------------------------------------------------------------------------
# Core capture routine
# --------------------------------------------------------------------------
def capture_pose_encodings(cap, voice: Voice, pose_label: str, instruction: str):
    """Runs a short live loop for one pose, returns a list of accepted
    128-d encodings for that pose."""
    voice.say(instruction)
    time.sleep(0.5)  # give the person a moment to move into position

    accepted = []
    attempts = 0

    while len(accepted) < FRAMES_TO_ACCEPT_PER_POSE and attempts < MAX_ATTEMPTS_PER_POSE:
        ret, frame = cap.read()
        if not ret:
            continue

        attempts += 1
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = apply_clahe(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(
            rgb, number_of_times_to_upsample=UPSAMPLE_TIMES, model=DETECTION_MODEL
        )

        # Reject frames with no face or more than one face — ambiguous data.
        if len(face_locations) != 1:
            _draw_status(frame, f"{pose_label}: faces detected = {len(face_locations)}")
            cv2.imshow("Registration", frame)
            cv2.waitKey(1)
            continue

        top, right, bottom, left = face_locations[0]
        face_width = right - left
        if face_width < MIN_FACE_WIDTH_PX:
            _draw_status(frame, f"{pose_label}: move closer")
            cv2.imshow("Registration", frame)
            cv2.waitKey(1)
            continue

        gray_crop = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        if gray_crop.size == 0:
            continue
        sharpness = sharpness_score(gray_crop)
        if sharpness < BLUR_VARIANCE_THRESHOLD:
            _draw_status(frame, f"{pose_label}: too blurry, hold still")
            cv2.imshow("Registration", frame)
            cv2.waitKey(1)
            continue

        encodings = face_recognition.face_encodings(
            rgb,
            known_face_locations=[face_locations[0]],
            num_jitters=REGISTRATION_JITTERS,
            model=ENCODING_MODEL,
        )
        if not encodings:
            continue

        accepted.append(encodings[0])

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 0), 2)
        _draw_status(frame, f"{pose_label}: captured {len(accepted)}/{FRAMES_TO_ACCEPT_PER_POSE}")
        cv2.imshow("Registration", frame)
        cv2.waitKey(1)

    if len(accepted) < FRAMES_TO_ACCEPT_PER_POSE:
        print(f"[WARN] Only captured {len(accepted)}/{FRAMES_TO_ACCEPT_PER_POSE} "
              f"valid samples for pose '{pose_label}' (hit attempt cap).")

    return accepted


def _draw_status(frame, text: str):
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Main registration flow
# --------------------------------------------------------------------------
def register_person():
    voice = Voice()
    name = input("Enter the person's name: ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    db = load_database()
    if name in db:
        overwrite = input(f"'{name}' already exists with {len(db[name])} "
                           f"encodings. Add more samples to it? (y/n): ").strip().lower()
        if overwrite != "y":
            print("Aborted.")
            return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    voice.say(f"Starting registration for {name}. Please follow the instructions.")

    all_new_encodings = []
    try:
        for pose_label, instruction in POSES:
            pose_encodings = capture_pose_encodings(cap, voice, pose_label, instruction)
            all_new_encodings.extend(pose_encodings)
            print(f"[INFO] Pose '{pose_label}': {len(pose_encodings)} encodings captured.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not all_new_encodings:
        voice.say("Registration failed. No valid face samples were captured.")
        return

    db.setdefault(name, [])
    db[name].extend(all_new_encodings)
    save_database(db)

    voice.say(f"Registration complete for {name}. "
              f"{len(all_new_encodings)} new samples saved, "
              f"{len(db[name])} total.")
    print(f"[INFO] Saved {len(all_new_encodings)} new encodings for '{name}'. "
          f"Total for this person: {len(db[name])}. Database: {ENCODINGS_PATH}")


if __name__ == "__main__":
    register_person()