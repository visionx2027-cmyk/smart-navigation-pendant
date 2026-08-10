import cv2
from ultralytics import YOLO
import sys
import os
import time
from collections import deque
from skimage.metrics import structural_similarity as ssim

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "camera_module"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "decision_engine"))

from capture import get_camera, get_frame
from voice import speak
from decision import get_distance_tier, build_message

model = YOLO("yolov8s.pt")
cam = get_camera(0)

SSIM_THRESHOLD = 0.90
FORCE_REFRESH_SECONDS = 0.7
LEFT_ZONE_END = 1 / 3
RIGHT_ZONE_START = 2 / 3
NAVIGATION_OBJECTS = ["person", "car", "bus", "motorcycle"]

prev_frame_gray = None
last_yolo_time = 0
detection_cache = {}

# Per-label smoothing history for box width (fixes distance jitter)
size_history = {}

# Per-label last spoken tag + time, so nav and other objects are
# tracked INDEPENDENTLY instead of one blocking the other
object_last_tag = {}
object_last_speak_time = {}

SPEAK_COOLDOWN = 3.0


def get_direction(center_x, frame_width):
    if center_x < frame_width * LEFT_ZONE_END:
        return "on your left"
    elif center_x > frame_width * RIGHT_ZONE_START:
        return "on your right"
    return "ahead"


def estimate_distance_m(smoothed_relative_size):
    if smoothed_relative_size > 0.5:
        return 0.4
    elif smoothed_relative_size > 0.25:
        return 1.5
    return 3.0


def run_yolo_and_update_cache(frame):
    results = model(frame, conf=0.6, verbose=False)
    boxes = results[0].boxes
    frame_width = frame.shape[1]
    seen_labels = set()

    for box in boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0]
        center_x = float((x1 + x2) / 2)
        box_width = float(x2 - x1)
        direction = get_direction(center_x, frame_width)
        relative_size = box_width / frame_width

        # Smooth over last 5 readings to kill jitter at threshold boundaries
        if label not in size_history:
            size_history[label] = deque(maxlen=5)
        size_history[label].append(relative_size)
        smoothed_size = sum(size_history[label]) / len(size_history[label])

        detection_cache[label] = {
            "direction": direction,
            "confidence": float(box.conf[0]),
            "smoothed_size": smoothed_size,
            "last_confirmed": time.time(),
        }
        seen_labels.add(label)

    for label in list(detection_cache.keys()):
        if label not in seen_labels:
            del detection_cache[label]
            size_history.pop(label, None)

    return results


def try_speak(label, tag, message, vibration=None):
    now = time.time()
    last_tag = object_last_tag.get(label, "")
    last_time = object_last_speak_time.get(label, 0)

    if tag != last_tag and (now - last_time) > SPEAK_COOLDOWN:
        if vibration:
            print(f"Speaking: {message} | vibration: {vibration}")
        else:
            print(f"Speaking: {message}")
        speak(message)
        object_last_tag[label] = tag
        object_last_speak_time[label] = now


print("Starting VISIONX object detection. Press 'q' to quit.")

while True:
    frame = get_frame(cam)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, (160, 90))
    current_time = time.time()

    should_run_yolo = (
        prev_frame_gray is None
        or ssim(prev_frame_gray, gray_small) < SSIM_THRESHOLD
        or (current_time - last_yolo_time) > FORCE_REFRESH_SECONDS
    )

    if should_run_yolo:
        results = run_yolo_and_update_cache(frame)
        last_yolo_time = current_time
        annotated_frame = results[0].plot()
    else:
        annotated_frame = frame

    prev_frame_gray = gray_small
    cv2.imshow("VISIONX Object Detection", annotated_frame)

    # Check EVERY detected label independently — nav objects don't block others
    for label, data in list(detection_cache.items()):
        direction = data["direction"]

        if label in NAVIGATION_OBJECTS:
            distance_m = estimate_distance_m(data["smoothed_size"])
            threshold, distance_label, vibration_intensity = get_distance_tier(distance_m)
            tag = f"{label}_{direction}_{distance_label}"
            message = build_message(label, direction, distance_label)
            try_speak(label, tag, message, vibration_intensity)
        else:
            tag = f"{label}_detected"
            message = f"{label} detected"
            try_speak(label, tag, message)

    # Clean up state for labels no longer in frame
    for label in list(object_last_tag.keys()):
        if label not in detection_cache:
            del object_last_tag[label]
            object_last_speak_time.pop(label, None)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()