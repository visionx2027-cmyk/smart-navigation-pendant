import cv2
import sys
import os
import time
import threading
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
from decision import get_distance_tier, build_message
from sensor import read_distance_real, cleanup as sensor_cleanup
from gps_reader import read_gps_simulated

import detector            # object_detection/detector.py
import recognizer          # face_recognition_module/recognizer.py
from reader import read_text_from_frame  # ocr_module/reader.py

# --- Config ---
SSIM_THRESHOLD = 0.90
FORCE_REFRESH_SECONDS = 0.7
FACE_EVERY_N_FRAMES = 2
SPEAK_COOLDOWN_GLOBAL = 1.0   # min gap between ANY two announcements

# --- State ---
prev_frame_gray = None
last_yolo_time = 0
frame_count = 0
last_speak_time = 0
last_gps = (None, None)
latest_distance_cm = 300

object_last_tag = {}
object_last_time = {}


def gps_background_loop():
    """GPS blocks ~1s internally, so it runs on its own thread and just
    updates a shared value — the main loop never waits on it."""
    global last_gps
    while True:
        last_gps = read_gps_simulated()
        time.sleep(1.5)


def try_speak_and_vibrate(tag, message, vibration_pattern="off"):
    global last_speak_time
    now = time.time()
    if (now - last_speak_time) > SPEAK_COOLDOWN_GLOBAL:
        vibrate(vibration_pattern)
        print(f"Speaking: {message} | vibration: {vibration_pattern}")
        speak(message)
        last_speak_time = now


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

            # --- Face recognition (every N frames) ---
            face_events = []
            if frame_count % FACE_EVERY_N_FRAMES == 0:
                face_events = recognizer.process_face_frame(frame, frame.shape[1])

            # ===================== DECISION LOGIC =====================
            distance_m = latest_distance_cm / 100
            threshold, distance_label, vibration_intensity = get_distance_tier(distance_m)

            spoken_this_cycle = False

            # Priority 1: obstacle danger zone always wins
            if distance_label == "STOP":
                try_speak_and_vibrate("obstacle_stop", "STOP. Obstacle very close.", "urgent")
                spoken_this_cycle = True

            # Priority 2: navigation objects (person/car/bus/motorcycle) combined with distance
            if not spoken_this_cycle:
                for event in detector.get_object_events():
                    if event["is_nav"]:
                        tier_threshold, tier_label, tier_vibe = get_distance_tier(event["distance_m"])
                        tag = f"{event['label']}_{event['direction']}_{tier_label}"
                        last_tag = object_last_tag.get(event["label"], "")
                        last_time = object_last_time.get(event["label"], 0)
                        if tag != last_tag and (current_time - last_time) > 3.0:
                            message = build_message(event["label"], event["direction"], tier_label)
                            try_speak_and_vibrate(tag, message, tier_vibe)
                            object_last_tag[event["label"]] = tag
                            object_last_time[event["label"]] = current_time
                            spoken_this_cycle = True
                            break
                    else:
                        tag = f"{event['label']}_detected"
                        last_tag = object_last_tag.get(event["label"], "")
                        last_time = object_last_time.get(event["label"], 0)
                        if tag != last_tag and (current_time - last_time) > 4.0:
                            try_speak_and_vibrate(tag, f"{event['label']} detected", "off")
                            object_last_tag[event["label"]] = tag
                            object_last_time[event["label"]] = current_time
                            spoken_this_cycle = True
                            break

            # Priority 3: face recognition announcements
            if not spoken_this_cycle:
                for fe in face_events:
                    if fe["should_announce"]:
                        if fe["name"] == "Unknown":
                            message = f"Unknown person {fe['direction']}"
                        else:
                            message = f"{fe['name']} is {fe['direction']}" if fe["direction"] == "center" else f"{fe['name']} is on your {fe['direction']}"
                        try_speak_and_vibrate(f"face_{fe['name']}_{fe['direction']}", message, "off")
                        spoken_this_cycle = True
                        break
            # ============================================================

            # --- OCR: on-demand only, since it's slow ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                print("Reading text...")
                text = read_text_from_frame(frame)
                if text:
                    speak(text)
                    print(f"OCR: {text}")
                else:
                    speak("No text found")
            elif key == ord('q'):
                break

            cv2.imshow("VISIONX", frame)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cam.release()
        cv2.destroyAllWindows()
        vibration_cleanup()
        sensor_cleanup()


if __name__ == "__main__":
    main()