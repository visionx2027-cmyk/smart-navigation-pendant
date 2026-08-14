"""
Extracted, importable core of object detection — no camera, no loop, no
speak() calls. object_detection/main.py is untouched and still works
standalone; this file is what root main.py imports instead.
"""
from ultralytics import YOLO
from collections import deque

model = YOLO("yolov8s.pt")

LEFT_ZONE_END = 1 / 3
RIGHT_ZONE_START = 2 / 3
NAVIGATION_OBJECTS = ["person", "car", "bus", "motorcycle", "bicycle"]

size_history = {}
detection_cache = {}


def get_direction(center_x, frame_width):
    # Raw left/center/right — matches recognizer.py's vocabulary so
    # main.py's decision layer can build every message the same way,
    # regardless of whether it came from face recognition or object
    # detection.
    if center_x < frame_width * LEFT_ZONE_END:
        return "left"
    elif center_x > frame_width * RIGHT_ZONE_START:
        return "right"
    return "center"


def run_yolo_and_update_cache(frame):
    results = model(frame, conf=0.35, verbose=False)
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

        if label not in size_history:
            size_history[label] = deque(maxlen=5)
        size_history[label].append(relative_size)
        smoothed_size = sum(size_history[label]) / len(size_history[label])

        detection_cache[label] = {
            "direction": direction,
            "confidence": float(box.conf[0]),
            "smoothed_size": smoothed_size,
        }
        seen_labels.add(label)

    for label in list(detection_cache.keys()):
        if label not in seen_labels:
            del detection_cache[label]
            size_history.pop(label, None)

    return results


def get_object_events():
    """Returns a list of dicts describing every currently cached object.
    NOTE: distance/"getting closer" is no longer derived here from bbox
    size -- that was the root cause of false approach warnings. Approach
    detection now lives entirely in main.py, driven by the ultrasonic
    sensor's rolling history. smoothed_size is kept in the cache only in
    case you want it later (e.g. scaling vibration intensity), but it no
    longer feeds any spoken message.
    """
    events = []
    for label, data in detection_cache.items():
        events.append({
            "label": label,
            "direction": data["direction"],
            "is_nav": label in NAVIGATION_OBJECTS,
        })
    return events