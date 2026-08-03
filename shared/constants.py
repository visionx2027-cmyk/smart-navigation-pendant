"""
Central place for values shared across multiple modules.
Changing a threshold here changes it everywhere — avoids magic numbers
scattered across the codebase.
"""

# --- Priority levels for the decision engine ---
# Lower number = higher priority (spoken first, interrupts lower priority speech)
PRIORITY_CRITICAL = 1   # obstacle very close, vehicle approaching
PRIORITY_MEDIUM = 2     # person, chair, door
PRIORITY_LOW = 3        # OCR, face recognition, GPS

# --- Distance thresholds (as a fraction of frame width, for camera-based estimation) ---
DISTANCE_VERY_CLOSE = 0.5
DISTANCE_NEARBY = 0.25

# --- Obstacle sensor thresholds (centimeters) ---
OBSTACLE_DANGER_CM = 30
OBSTACLE_WARNING_CM = 100

# --- Cooldown windows (seconds) — prevents repeated announcements ---
COOLDOWN_OBJECT = 4
COOLDOWN_OBSTACLE = 3
COOLDOWN_FACE = 5
COOLDOWN_GPS = 10

# --- Object classes considered "vehicles" for critical-priority handling ---
VEHICLE_CLASSES = {"car", "bus", "motorcycle", "truck"}

# --- Object classes the system actually cares about announcing ---
# (filters YOLO's full 80-class COCO output down to what's useful here)
RELEVANT_OBJECT_CLASSES = {
    "person", "chair", "bottle", "cell phone", "laptop",
    "car", "bus", "motorcycle", "traffic light", "stop sign", "door"
}