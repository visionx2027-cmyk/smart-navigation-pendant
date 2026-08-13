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
NAVIGATION_OBJECTS = ["person", "car", "bus", "motorcycle"]

size_history = {}
detection_cache = {}


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
    """Returns a list of dicts describing every currently cached object."""
    events = []
    for label, data in detection_cache.items():
        if label in NAVIGATION_OBJECTS:
            distance_m = estimate_distance_m(data["smoothed_size"])
            events.append({"label": label, "direction": data["direction"],
                            "distance_m": distance_m, "is_nav": True})
        else:
            events.append({"label": label, "direction": data["direction"],
                            "distance_m": None, "is_nav": False})
    return events