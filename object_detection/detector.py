import cv2
from ultralytics import YOLO
import sys
import os
import time
from collections import deque

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "camera_module"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
from capture import get_camera, get_frame
from voice import speak

model = YOLO("yolov8s.pt")
cam = get_camera(0)

cooldown = 6
last_tag = ""
last_speak_time = 0

# Objects that get full motion tracking (approaching/receding/moving)
# Everything else only gets direction + distance, no motion state
MOTION_TRACKED_OBJECTS = ["person", "car"]

# Per-object rolling history for smoothing + motion detection
# Keyed by label since we only track one instance of each tracked object at a time
history = {}
for obj in MOTION_TRACKED_OBJECTS:
    history[obj] = {
        "size": deque(maxlen=6),
        "center_x": deque(maxlen=6),
    }


def get_direction(center_x, frame_width):
    frame_center = frame_width / 2
    if center_x < frame_center - 80:
        return "on your left"
    elif center_x > frame_center + 80:
        return "on your right"
    return "ahead"


def get_distance_word(relative_size):
    if relative_size > 0.5:
        return "very close"
    elif relative_size > 0.25:
        return "nearby"
    return "far"


def get_motion_state(obj_history, current_size, current_center_x):
    """
    Compares the current reading against the SMOOTHED AVERAGE of recent
    frames (not the single previous frame) to avoid reacting to jitter.
    Returns None if there isn't enough history yet, or if the change
    isn't meaningful enough to call it real motion.
    """
    size_hist = obj_history["size"]
    center_hist = obj_history["center_x"]

    if len(size_hist) < 6:
        return None  # not enough data yet to judge motion reliably

    avg_size = sum(size_hist) / len(size_hist)
    avg_center = sum(center_hist) / len(center_hist)

    size_diff = current_size - avg_size
    center_diff = current_center_x - avg_center

    # Thresholds tuned to ignore natural sway/breathing jitter
    SIZE_THRESHOLD = 0.10
    CENTER_THRESHOLD = 90

    # Size change matters more than lateral change when both are present,
    # since "approaching" is usually more urgent than "moving sideways"
    if abs(size_diff) > SIZE_THRESHOLD:
        return "approaching" if size_diff > 0 else "moving away"
    elif abs(center_diff) > CENTER_THRESHOLD:
        return "moving left" if center_diff < 0 else "moving right"

    return "stationary"


while True:
    frame = get_frame(cam)
    results = model(frame, conf=0.6, verbose=False)
    annotated_frame = results[0].plot()
    cv2.imshow("VISIONX Object Detection", annotated_frame)

    boxes = results[0].boxes
    current_time = time.time()

    if len(boxes) > 0:
        top_box = boxes[boxes.conf.argmax()]
        class_id = int(top_box.cls[0])
        label = model.names[class_id]

        x1, y1, x2, y2 = top_box.xyxy[0]
        box_width = float(x2 - x1)
        box_center_x = float((x1 + x2) / 2)
        frame_width = frame.shape[1]
        relative_size = box_width / frame_width

        direction = get_direction(box_center_x, frame_width)
        distance = get_distance_word(relative_size)

        motion = None
        if label in MOTION_TRACKED_OBJECTS:
            obj_history = history[label]
            motion = get_motion_state(obj_history, relative_size, box_center_x)

            # Update history AFTER computing motion, so this frame's
            # reading becomes part of next frame's baseline
            obj_history["size"].append(relative_size)
            obj_history["center_x"].append(box_center_x)

        # Build the composite state tag — speech only fires when THIS changes
        tag = f"{label}_{direction}_{distance}_{motion}"

        if tag != last_tag or (current_time - last_speak_time) > 15:
            # Decide phrasing: lead with the most urgent/relevant piece
            if motion == "approaching":
                message = f"{label} approaching"
            elif motion == "moving away":
                message = f"{label} moving away"
            elif motion in ("moving left", "moving right"):
                message = f"{label} {motion}"
            elif distance == "very close":
                message = f"{label} very close"
            else:
                message = f"{label} {direction}"

            print(f"Speaking: {message}")
            speak(message)
            last_tag = tag
            last_speak_time = current_time

    else:
        last_tag = ""
        for obj in MOTION_TRACKED_OBJECTS:
            history[obj]["size"].clear()
            history[obj]["center_x"].clear()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()