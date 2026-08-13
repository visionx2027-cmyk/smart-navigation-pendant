"""
Extracted, importable core of face recognition — no camera, no loop.
face_recognition_module/main.py is untouched and still works standalone.
"""
import os, cv2, time, pickle, numpy as np, face_recognition
from collections import deque, Counter

KNOWN_FACES_DIR = "face_recognition_module/known_faces"
ENCODINGS_PATH = os.path.join(KNOWN_FACES_DIR, "encodings.pkl")

TOLERANCE = 0.58
K_NEIGHBORS = 7
MIN_VOTE_FRACTION = 0.55
MARGIN_THRESHOLD = 0.03
SMOOTHING_WINDOW = 6
CONSISTENCY_REQUIRED = 4
ANNOUNCE_COOLDOWN_SEC = 8.0
DIRECTION_LEFT_BOUND = 0.40
DIRECTION_RIGHT_BOUND = 0.60
DIRECTION_SMOOTHING_WINDOW = 4
DIRECTION_CONSISTENCY_REQUIRED = 3
RECOGNITION_JITTERS = 1
DETECTION_MODEL = "hog"
UPSAMPLE_TIMES = 1
ENCODING_MODEL = "large"
EXPECTED_ENCODING_LENGTH = 128


def apply_clahe(bgr_frame):
    lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    merged = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def load_database():
    if not os.path.exists(ENCODINGS_PATH):
        raise FileNotFoundError(f"No encodings at {ENCODINGS_PATH}. Run register_face.py first.")
    with open(ENCODINGS_PATH, "rb") as f:
        db = pickle.load(f)
    names, encodings = [], []
    for name, enc_list in db.items():
        if isinstance(enc_list, np.ndarray) and enc_list.ndim == 1:
            enc_list = [enc_list]
        for enc in enc_list:
            arr = np.asarray(enc, dtype=np.float64)
            if arr.ndim == 1 and arr.shape[0] == EXPECTED_ENCODING_LENGTH:
                names.append(name)
                encodings.append(arr)
    if not encodings:
        raise ValueError(f"No valid encodings in {ENCODINGS_PATH}.")
    return np.array(names), np.stack(encodings)


def identify_face(face_encoding, known_names, known_encodings):
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    eligible_idx = np.where(distances <= TOLERANCE)[0]
    if eligible_idx.size == 0:
        return None
    eligible_sorted = eligible_idx[np.argsort(distances[eligible_idx])]
    top_k_idx = eligible_sorted[:K_NEIGHBORS]
    top_k_names = known_names[top_k_idx]
    vote_counts = Counter(top_k_names)
    winner, winner_votes = vote_counts.most_common(1)[0]
    if winner_votes / len(top_k_idx) < MIN_VOTE_FRACTION:
        return None
    winner_best_dist = float(np.min(distances[known_names == winner]))
    other_mask = known_names != winner
    if np.any(other_mask):
        runner_up_best_dist = float(np.min(distances[other_mask]))
        if (runner_up_best_dist - winner_best_dist) < MARGIN_THRESHOLD:
            return None
    return winner


def estimate_direction(left, right, frame_width):
    fraction = ((left + right) / 2.0) / frame_width
    if fraction < DIRECTION_LEFT_BOUND:
        return "left"
    elif fraction > DIRECTION_RIGHT_BOUND:
        return "right"
    return "center"


class FaceTrack:
    def __init__(self, track_id, center):
        self.id = track_id
        self.center = center
        self.history = deque(maxlen=SMOOTHING_WINDOW)
        self.direction_history = deque(maxlen=DIRECTION_SMOOTHING_WINDOW)
        self.last_announced = None
        self.last_announced_time = 0.0

    def update_center(self, center): self.center = center
    def push_result(self, name): self.history.append(name)
    def push_direction(self, d): self.direction_history.append(d)

    def stable_name(self):
        if len(self.history) < CONSISTENCY_REQUIRED:
            return None
        name, count = Counter(self.history).most_common(1)[0]
        return name if name is not None and count >= CONSISTENCY_REQUIRED else None

    def stable_direction(self):
        if not self.direction_history:
            return "center"
        direction, count = Counter(self.direction_history).most_common(1)[0]
        return direction if count >= DIRECTION_CONSISTENCY_REQUIRED else self.direction_history[-1]

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
    def __init__(self, max_distance_px=80):
        self.tracks = {}
        self.next_id = 0
        self.max_distance_px = max_distance_px

    def match(self, centers):
        assigned = {}
        unused = set(self.tracks.keys())
        for i, center in enumerate(centers):
            best_id, best_dist = None, None
            for tid in unused:
                d = np.hypot(center[0] - self.tracks[tid].center[0], center[1] - self.tracks[tid].center[1])
                if best_dist is None or d < best_dist:
                    best_dist, best_id = d, tid
            if best_id is not None and best_dist <= self.max_distance_px:
                self.tracks[best_id].update_center(center)
                assigned[i] = self.tracks[best_id]
                unused.discard(best_id)
            else:
                new_track = FaceTrack(self.next_id, center)
                self.tracks[self.next_id] = new_track
                assigned[i] = new_track
                self.next_id += 1
        for tid in list(self.tracks.keys()):
            if self.tracks[tid] not in assigned.values():
                del self.tracks[tid]
        return assigned


_known_names, _known_encodings = load_database()
_tracker = SimpleTracker()


def process_face_frame(frame, frame_width=640):
    """Returns list of dicts: {name_or_Unknown, direction, should_announce}."""
    processed = apply_clahe(frame)
    rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb, number_of_times_to_upsample=UPSAMPLE_TIMES, model=DETECTION_MODEL)
    face_encodings = face_recognition.face_encodings(rgb, known_face_locations=face_locations, num_jitters=RECOGNITION_JITTERS, model=ENCODING_MODEL)
    centers = [((l + r) // 2, (t + b) // 2) for (t, r, b, l) in face_locations]
    tracks = _tracker.match(centers)

    events = []
    for i, face_encoding in enumerate(face_encodings):
        name = identify_face(face_encoding, _known_names, _known_encodings)
        track = tracks[i]
        track.push_result(name)
        top, right, bottom, left = face_locations[i]
        direction = estimate_direction(left, right, frame_width)
        track.push_direction(direction)
        stable_direction = track.stable_direction()
        stable_name = track.stable_name()

        announce = False
        if stable_name and track.should_announce(stable_name, stable_direction):
            track.mark_announced(stable_name, stable_direction)
            announce = True

        events.append({
            "name": stable_name if stable_name else "Unknown",
            "direction": stable_direction,
            "should_announce": announce,
        })
    return events