import cv2
import sys
import os
import time
import threading
from collections import deque
from skimage.metrics import structural_similarity as ssim

sys.path.append(os.path.join(os.path.dirname(__file__), "camera_module"))
sys.path.append(os.path.join(os.path.dirname(__file__), "shared"))
sys.path.append(os.path.join(os.path.dirname(__file__), "decision_engine"))
sys.path.append(os.path.join(os.path.dirname(__file__), "obstacle_detection"))
sys.path.append(os.path.join(os.path.dirname(__file__), "gps_module"))
sys.path.append(os.path.join(os.path.dirname(__file__), "object_detection"))
sys.path.append(os.path.join(os.path.dirname(__file__), "face_recognition_module"))
sys.path.append(os.path.join(os.path.dirname(__file__), "ocr_module"))

from capture import get_camera, get_frame
from voice import speak
from vibration import vibrate, cleanup as vibration_cleanup
from sensor import read_distance_real, cleanup as sensor_cleanup
from gps_reader import read_gps_simulated

import detector            # object_detection/detector.py
import recognizer          # face_recognition_module/recognizer.py
from reader import read_text_from_frame  # ocr_module/reader.py

# --- Config ---
SSIM_THRESHOLD = 0.90
FORCE_REFRESH_SECONDS = 0.7
FACE_EVERY_N_FRAMES = 2
SPEAK_COOLDOWN_GLOBAL = 1.0     # min gap between ANY two announcements

# Ultrasonic trend / "getting closer" config.
# sensor.py returns a single raw reading with no history or filtering, so
# all trend detection has to happen here. We keep a short rolling window
# of readings and only call it "approaching" if there's a real, sustained
# decrease across the window — not a single low reading, not noise.
DISTANCE_HISTORY_LEN = 5
APPROACH_DROP_THRESHOLD_CM = 20   # total drop required across the window
NOISE_TOLERANCE_CM = 2            # per-step wobble that doesn't count against "decreasing"
OBSTACLE_STOP_CM = 50             # hard danger zone, highest priority

DIRECTION_PHRASE = {"left": "on left", "right": "on right", "center": "in front"}

# --- State ---
prev_frame_gray = None
last_yolo_time = 0
frame_count = 0
last_speak_time = 0
last_speak_tag = None
last_gps = (None, None)
latest_distance_cm = 300
distance_history = deque(maxlen=DISTANCE_HISTORY_LEN)


def gps_background_loop():
    """GPS blocks ~1s internally, so it runs on its own thread and just
    updates a shared value — the main loop never waits on it."""
    global last_gps
    while True:
        last_gps = read_gps_simulated()
        time.sleep(1.5)


def is_approaching(history):
    """True only if the ultrasonic readings show a real, sustained
    decreasing trend — not one low reading, not sensor noise, not a
    person standing still or walking away."""
    if len(history) < DISTANCE_HISTORY_LEN:
        return False

    values = list(history)
    net_drop = values[0] - values[-1]
    if net_drop < APPROACH_DROP_THRESHOLD_CM:
        return False

    decreasing_steps = sum(
        1 for i in range(1, len(values))
        if values[i] < values[i - 1] - NOISE_TOLERANCE_CM
    )
    # require most steps to actually be decreasing, not just the endpoints
    return decreasing_steps >= (len(values) - 2)


def try_speak_and_vibrate(tag, message, vibration_pattern="off"):
    global last_speak_time, last_speak_tag
    now = time.time()
    if (now - last_speak_time) > SPEAK_COOLDOWN_GLOBAL and tag != last_speak_tag:
        vibrate(vibration_pattern)
        print(f"Speaking: {message} | vibration: {vibration_pattern}")
        speak(message)
        last_speak_time = now
        last_speak_tag = tag
    elif (now - last_speak_time) > SPEAK_COOLDOWN_GLOBAL:
        # same tag as last time but cooldown has expired long enough that
        # it's a legitimate repeat (e.g. object is still there) -- still
        # gate it so we don't spam every single frame
        vibrate(vibration_pattern)
        print(f"Speaking: {message} | vibration: {vibration_pattern}")
        speak(message)
        last_speak_time = now
        last_speak_tag = tag


def build_final_message(face_events, object_events, approaching, distance_cm):
    """Single decision layer. Returns (message, tag) or (None, None).
    Priority: obstacle STOP > known person > unknown moving person >
    other moving object > static object > ultrasonic approach warning.
    """
    # Priority 0: hard obstacle danger zone always wins
    if distance_cm is not None and distance_cm <= OBSTACLE_STOP_CM:
        return f"Stop, obstacle very close, {int(distance_cm)} centimeters", "obstacle_stop"

    known_person_present = any(fe["name"] != "Unknown" for fe in face_events)
    face_seen_this_frame = len(face_events) > 0

    # Priority 1: known person
    for fe in face_events:
        if fe["name"] != "Unknown" and fe["should_announce"]:
            phrase = DIRECTION_PHRASE.get(fe["direction"], "nearby")
            return f"{fe['name']} is {phrase}", f"face_{fe['name']}_{fe['direction']}"

    # Priority 2: unknown moving person
    for fe in face_events:
        if fe["name"] == "Unknown" and fe["should_announce"]:
            phrase = DIRECTION_PHRASE.get(fe["direction"], "nearby")
            return f"Person is {phrase}", f"face_unknown_{fe['direction']}"

    # Priority 3/4: object-detector results.
    # If any face box was seen this frame (known or unknown), the face
    # pipeline already owns the "person" announcement -- suppress the
    # generic object-detector "person" so it can't override or duplicate
    # the name/direction the face module already produced. This is the
    # fix for known people being reported as generic "person": previously
    # suppression only applied on the exact frame the face module chose
    # to announce, so every other frame let the object detector win.
    moving_events = [
        e for e in object_events
        if e["is_nav"] and not (e["label"] == "person" and face_seen_this_frame)
    ]
    static_events = [e for e in object_events if not e["is_nav"]]

    if moving_events:
        e = moving_events[0]
        phrase = DIRECTION_PHRASE.get(e["direction"], "nearby")
        return f"{e['label']} is {phrase}", f"obj_{e['label']}_{e['direction']}"

    if static_events:
        e = static_events[0]
        # Static objects: name only, never a direction (per spec).
        return f"{e['label']} detected", f"obj_{e['label']}_static"

    # Priority 5: ultrasonic-confirmed approach warning with nothing else to say
    if approaching:
        return f"Obstacle getting closer, {int(distance_cm)} centimeters", "ultrasonic_approach"

    return None, None


def main():
    global prev_frame_gray, last_yolo_time, frame_count, latest_distance_cm

    cam = get_camera(0)
    gps_thread = threading.Thread(target=gps_background_loop, daemon=True)
    gps_thread.start()

    print("VISIONX integrated system running. Press 'r' for OCR, 'q' to quit.")

    try:
        while True:
            frame = get_frame(cam)
            frame_count += 1
            current_time = time.time()

            # --- Obstacle distance: fast, safe to call every loop ---
            distance = read_distance_real()
            if distance is not None:
                latest_distance_cm = distance
                distance_history.append(distance)

            approaching = is_approaching(distance_history)

            # --- SSIM gate for YOLO ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (160, 90))
            should_run_yolo = (
                prev_frame_gray is None
                or ssim(prev_frame_gray, gray_small) < SSIM_THRESHOLD
                or (current_time - last_yolo_time) > FORCE_REFRESH_SECONDS
            )
            if should_run_yolo:
                detector.run_yolo_and_update_cache(frame)
                last_yolo_time = current_time
            prev_frame_gray = gray_small

            # --- Face recognition (every N frames; independent of SSIM/YOLO) ---
            face_events = []
            if frame_count % FACE_EVERY_N_FRAMES == 0:
                face_events = recognizer.process_face_frame(frame, frame.shape[1])

            object_events = detector.get_object_events()

            # ===================== SINGLE DECISION LAYER =====================
            message, tag = build_final_message(
                face_events, object_events, approaching, latest_distance_cm
            )
            if message:
                try_speak_and_vibrate(tag, message)
            # ===================================================================

            # --- OCR: on-demand only, wired separately (slow, not per-frame) ---
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cam.stop()
        vibration_cleanup()
        sensor_cleanup()


if __name__ == "__main__":
    main()