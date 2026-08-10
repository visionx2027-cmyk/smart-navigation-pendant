"""
main.py
-------
Smart Blind Navigation Pendant — Real-Time Face Recognition Module

Architecture notes (mirrors the fixes made in register_face.py):

1. MATCHING STRATEGY: For each detected face we compute distances to
   EVERY stored encoding of EVERY person (not one encoding per person).
   We then take the k nearest neighbors overall and let them vote on
   identity, breaking ties by lowest average distance. This is what
   actually lets a 20-30 degree turn get recognized: if the person's
   "Turn Left" registration samples are close to the current frame,
   they win the vote even though their "Look Straight" sample might be
   far. A single "closest match" against averaged/blended encodings
   (the likely previous design) throws that signal away.

2. TOLERANCE: 0.60 as a hard cutoff on a single distance is fragile.
   Here tolerance is used as a filter on which neighbors are even
   eligible to vote (default 0.58, slightly tighter than 0.60 to avoid
   false accepts now that we have far more candidate encodings per
   person to compare against), and identity is decided by voting
   + margin, not by the raw closest distance alone.

3. MARGIN CHECK: if the best-matching person and the second-best
   person are separated by too small a distance margin, we treat the
   result as ambiguous rather than risking a wrong announcement.

4. ENCODING MODEL MATCHES REGISTRATION: model="large" is used here too
   — mismatched landmark models between registration and recognition
   would reintroduce the alignment problem.

5. TEMPORAL SMOOTHING: a single frame's recognition can flicker
   (motion blur, brief occlusion). We keep a short rolling history per
   face track and only announce once a label is consistent across
   several consecutive frames, with a cooldown so the same person
   isn't announced every frame.

6. NON-BLOCKING VOICE: TTS runs on a background thread so it never
   stalls the camera loop (important for a real-time navigation aid).
"""

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
ANNOUNCE_COOLDOWN_SEC = 8.0     # don't re-announce the same name+direction too often

# Direction zones as a fraction of frame width. A face center within the
# middle band is "in front of you"; outside it, "left"/"right". Assumes the
# camera faces the same direction as the wearer (not a mirrored selfie feed) —
# left in the frame = left of the person wearing the pendant.
DIRECTION_LEFT_BOUND = 0.40
DIRECTION_RIGHT_BOUND = 0.60
DIRECTION_SMOOTHING_WINDOW = 4
DIRECTION_CONSISTENCY_REQUIRED = 3

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
EXPECTED_ENCODING_LENGTH = 128


def load_database():
    """Returns (names_array, encodings_matrix) flattened for vectorized
    distance computation. names_array[i] corresponds to encodings_matrix[i].

    Defensive against malformed/legacy pickle contents: some earlier version
    of the registration script may have stored a single flat encoding per
    name instead of a list of encodings, or a run may have been interrupted
    mid-save. Rather than crash on np.array() with an inhomogeneous shape,
    we validate every entry and skip anything that isn't a clean 128-d
    vector, warning about what got skipped."""
    if not os.path.exists(ENCODINGS_PATH):
        raise FileNotFoundError(
            f"No encodings found at {ENCODINGS_PATH}. Run register_face.py first."
        )
    with open(ENCODINGS_PATH, "rb") as f:
        db = pickle.load(f)

    names = []
    encodings = []
    skipped = 0

    for name, enc_list in db.items():
        # Legacy format safety: if enc_list is actually a single encoding
        # (a flat array/list of length 128) rather than a list of encodings,
        # wrap it so it's iterated correctly instead of iterating floats.
        if isinstance(enc_list, np.ndarray) and enc_list.ndim == 1:
            enc_list = [enc_list]

        for enc in enc_list:
            arr = np.asarray(enc, dtype=np.float64)
            if arr.ndim == 1 and arr.shape[0] == EXPECTED_ENCODING_LENGTH:
                names.append(name)
                encodings.append(arr)
            else:
                skipped += 1

    if skipped:
        print(f"[WARN] Skipped {skipped} malformed encoding entr"
              f"{'y' if skipped == 1 else 'ies'} in {ENCODINGS_PATH} "
              f"(wrong shape — likely from an old/incompatible registration run). "
              f"Consider deleting the file and re-running register_face.py "
              f"if this number looks large.")

    if not encodings:
        raise ValueError(
            f"No valid encodings found in {ENCODINGS_PATH}. "
            f"Delete it and run register_face.py again."
        )

    return np.array(names), np.stack(encodings)


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
# Direction estimation: where is this face relative to the wearer?
#
# We use the horizontal position of the face's bounding-box center within
# the frame, not just the raw center point, so it's independent of how
# SimpleTracker groups detections. Assumes the camera faces outward in the
# same direction the wearer is facing (see config note above) — if you
# later add a wide-angle or multi-camera setup this mapping will need to
# change accordingly.
# --------------------------------------------------------------------------
def estimate_direction(left, right, frame_width):
    face_center_x = (left + right) / 2.0
    fraction = face_center_x / frame_width
    if fraction < DIRECTION_LEFT_BOUND:
        return "left"
    elif fraction > DIRECTION_RIGHT_BOUND:
        return "right"
    return "center"


def direction_phrase(name, direction):
    if direction == "left":
        return f"{name} is on your left"
    elif direction == "right":
        return f"{name} is on your right"
    return f"{name} is in front of you"


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
        self.direction_history = deque(maxlen=DIRECTION_SMOOTHING_WINDOW)
        self.last_announced = None  # (name, direction) tuple
        self.last_announced_time = 0.0

    def update_center(self, center):
        self.center = center

    def push_result(self, name):
        self.history.append(name)

    def push_direction(self, direction):
        self.direction_history.append(direction)

    def stable_name(self):
        if len(self.history) < CONSISTENCY_REQUIRED:
            return None
        counts = Counter(self.history)
        name, count = counts.most_common(1)[0]
        if name is not None and count >= CONSISTENCY_REQUIRED:
            return name
        return None

    def stable_direction(self):
        """Majority direction over a short window. Falls back to the most
        recent single reading if there isn't enough history yet or the
        window is split — direction should stay responsive since the
        person may be walking past, unlike identity which we want to be
        conservative about."""
        if not self.direction_history:
            return "center"
        counts = Counter(self.direction_history)
        direction, count = counts.most_common(1)[0]
        if count >= DIRECTION_CONSISTENCY_REQUIRED:
            return direction
        return self.direction_history[-1]

    def should_announce(self, name, direction):
        now = time.time()
        current = (name, direction)
        if current == self.last_announced and (now - self.last_announced_time) < ANNOUNCE_COOLDOWN_SEC:
            return False
        return True

    def mark_announced(self, name, direction):
        self.last_announced = (name, direction)
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

                    top, right, bottom, left = face_locations[i]
                    direction = estimate_direction(left, right, FRAME_WIDTH)
                    track.push_direction(direction)
                    stable_direction = track.stable_direction()

                    stable_name = track.stable_name()
                    display_label = stable_name if stable_name else "Analyzing..."

                    if stable_name and track.should_announce(stable_name, stable_direction):
                        voice.say_async(direction_phrase(stable_name, stable_direction))
                        track.mark_announced(stable_name, stable_direction)

                    dist_str = f"{distance:.3f}" if distance is not None else "n/a"
                    label_text = display_label if stable_name else "Unknown"
                    labels.append(f"{label_text} ({dist_str}) [{stable_direction}]")

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