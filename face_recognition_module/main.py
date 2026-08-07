
import os
import cv2
import time
import pickle
import threading
import numpy as np
import face_recognition
from collections import deque, Counter
import pyttsx3

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_PATH = os.path.join(KNOWN_FACES_DIR, "encodings.pkl")

ENCODING_MODEL = "large"        # must match register_face.py
RECOGNITION_JITTERS = 1         # real-time: speed matters, 1 is fine here
DETECTION_MODEL = "hog"         # switch to "cnn" only if you have GPU/Pi headroom
UPSAMPLE_TIMES = 1

TOLERANCE = 0.58                # eligibility cutoff for a neighbor to vote
K_NEIGHBORS = 7                 # how many nearest encodings vote on identity
MIN_VOTE_FRACTION = 0.55        # winning name must hold at least this share of votes
MARGIN_THRESHOLD = 0.03         # min distance gap vs runner-up to accept

SMOOTHING_WINDOW = 6            # frames of history per track
CONSISTENCY_REQUIRED = 4        # how many of the last N frames must agree
ANNOUNCE_COOLDOWN_SEC = 8.0     # don't re-announce the same name too often

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
PROCESS_EVERY_N_FRAMES = 2      # skip frames for detection to keep FPS up on Pi


# --------------------------------------------------------------------------
# Voice (non-blocking)
# --------------------------------------------------------------------------
class Voice:
    def __init__(self):
        self._lock = threading.Lock()

    def say_async(self, text: str):
        threading.Thread(target=self._speak, args=(text,), daemon=True).start()

    def _speak(self, text: str):
        with self._lock:
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            print(f"[VOICE] {text}")
            engine.say(text)
            engine.runAndWait()
            engine.stop()


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
def load_database():
    """Returns (names_array, encodings_matrix) flattened for vectorized
    distance computation. names_array[i] corresponds to encodings_matrix[i]."""
    if not os.path.exists(ENCODINGS_PATH):
        raise FileNotFoundError(
            f"No encodings found at {ENCODINGS_PATH}. Run register_face.py first."
        )
    with open(ENCODINGS_PATH, "rb") as f:
        db = pickle.load(f)

    names = []
    encodings = []
    for name, enc_list in db.items():
        for enc in enc_list:
            names.append(name)
            encodings.append(enc)

    if not encodings:
        raise ValueError("Encoding database is empty. Register at least one person.")

    return np.array(names), np.array(encodings)


# --------------------------------------------------------------------------
# Preprocessing (must match register_face.py's lighting normalization)
# --------------------------------------------------------------------------
def apply_clahe(bgr_frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    merged = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------
# Identity decision: k-NN vote + margin check
# --------------------------------------------------------------------------
def identify_face(face_encoding, known_names, known_encodings):
    """Returns (name_or_None, best_distance). None means Unknown / ambiguous."""
    distances = face_recognition.face_distance(known_encodings, face_encoding)

    # Only encodings within tolerance are eligible to vote at all.
    eligible_idx = np.where(distances <= TOLERANCE)[0]
    if eligible_idx.size == 0:
        return None, float(np.min(distances)) if distances.size else None

    # Take the k nearest among eligible candidates.
    eligible_sorted = eligible_idx[np.argsort(distances[eligible_idx])]
    top_k_idx = eligible_sorted[:K_NEIGHBORS]
    top_k_names = known_names[top_k_idx]
    top_k_distances = distances[top_k_idx]

    vote_counts = Counter(top_k_names)
    winner, winner_votes = vote_counts.most_common(1)[0]
    vote_fraction = winner_votes / len(top_k_idx)

    if vote_fraction < MIN_VOTE_FRACTION:
        return None, float(np.min(distances))  # too split between candidates

    # Margin check: compare winner's best distance to the best distance
    # among all OTHER names in the full candidate pool (not just top-k),
    # so a close second-best person can't sneak through.
    winner_best_dist = float(np.min(distances[known_names == winner]))
    other_mask = known_names != winner
    if np.any(other_mask):
        runner_up_best_dist = float(np.min(distances[other_mask]))
        if (runner_up_best_dist - winner_best_dist) < MARGIN_THRESHOLD:
            return None, winner_best_dist  # too close to call

    return winner, winner_best_dist


# --------------------------------------------------------------------------
# Simple centroid-based tracker so smoothing survives minor frame-to-frame
# movement of the same face (good enough for a single-user pendant; for
# multi-face scenes each track gets its own smoothing history).
# --------------------------------------------------------------------------
class FaceTrack:
    def __init__(self, track_id, center):
        self.id = track_id
        self.center = center
        self.history = deque(maxlen=SMOOTHING_WINDOW)
        self.last_announced = None
        self.last_announced_time = 0.0

    def update_center(self, center):
        self.center = center

    def push_result(self, name):
        self.history.append(name)

    def stable_name(self):
        if len(self.history) < CONSISTENCY_REQUIRED:
            return None
        counts = Counter(self.history)
        name, count = counts.most_common(1)[0]
        if name is not None and count >= CONSISTENCY_REQUIRED:
            return name
        return None

    def should_announce(self, name):
        now = time.time()
        if name == self.last_announced and (now - self.last_announced_time) < ANNOUNCE_COOLDOWN_SEC:
            return False
        return True

    def mark_announced(self, name):
        self.last_announced = name
        self.last_announced_time = time.time()


class SimpleTracker:
    """Matches new detections to existing tracks by nearest centroid."""

    def __init__(self, max_distance_px=80):
        self.tracks = {}
        self.next_id = 0
        self.max_distance_px = max_distance_px

    def match(self, centers):
        assigned = {}
        unused_track_ids = set(self.tracks.keys())

        for i, center in enumerate(centers):
            best_id, best_dist = None, None
            for tid in unused_track_ids:
                d = np.hypot(center[0] - self.tracks[tid].center[0],
                             center[1] - self.tracks[tid].center[1])
                if best_dist is None or d < best_dist:
                    best_dist, best_id = d, tid

            if best_id is not None and best_dist <= self.max_distance_px:
                self.tracks[best_id].update_center(center)
                assigned[i] = self.tracks[best_id]
                unused_track_ids.discard(best_id)
            else:
                new_track = FaceTrack(self.next_id, center)
                self.tracks[self.next_id] = new_track
                assigned[i] = new_track
                self.next_id += 1

        # Drop stale tracks that weren't matched this frame.
        for tid in list(self.tracks.keys()):
            if self.tracks[tid] not in assigned.values():
                del self.tracks[tid]

        return assigned


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def run():
    voice = Voice()
    known_names, known_encodings = load_database()
    print(f"[INFO] Loaded {len(known_encodings)} encodings across "
          f"{len(set(known_names))} people.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    tracker = SimpleTracker()
    frame_count = 0
    last_face_locations = []
    last_labels = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame_count += 1

            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                processed = apply_clahe(frame)
                rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(
                    rgb, number_of_times_to_upsample=UPSAMPLE_TIMES, model=DETECTION_MODEL
                )
                face_encodings = face_recognition.face_encodings(
                    rgb, known_face_locations=face_locations,
                    num_jitters=RECOGNITION_JITTERS, model=ENCODING_MODEL
                )

                centers = [
                    ((left + right) // 2, (top + bottom) // 2)
                    for (top, right, bottom, left) in face_locations
                ]
                tracks = tracker.match(centers)

                labels = []
                for i, face_encoding in enumerate(face_encodings):
                    name, distance = identify_face(face_encoding, known_names, known_encodings)
                    track = tracks[i]
                    track.push_result(name)

                    stable = track.stable_name()
                    display_label = stable if stable else "Analyzing..."

                    if stable and stable != "Unknown" and track.should_announce(stable):
                        voice.say_async(f"{stable} is in front of you")
                        track.mark_announced(stable)

                    dist_str = f"{distance:.3f}" if distance is not None else "n/a"
                    labels.append(f"{display_label if stable else 'Unknown'} ({dist_str})")

                last_face_locations = face_locations
                last_labels = labels

            # Draw the most recent detection results every frame (even on
            # skipped frames) so the video doesn't look frozen.
            for (top, right, bottom, left), label in zip(last_face_locations, last_labels):
                color = (0, 200, 0) if "Unknown" not in label else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, label, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            cv2.imshow("VISIONX - Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()