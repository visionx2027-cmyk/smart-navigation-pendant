"""
Decision Engine — combines what YOLO detected (label, direction) with
distance to decide the escalation tier and what message/vibration to trigger.
Does not run YOLO, does not read any sensor itself — pure decision logic.
"""

DISTANCE_TIERS = [
    (0.5, "STOP", "urgent"),
    (1.0, "very close", "strong"),
    (2.0, "nearby", "normal"),
    (3.0, "ahead", "small"),
]


def get_distance_tier(distance_m):
    for threshold, label, vibration in DISTANCE_TIERS:
        if distance_m <= threshold:
            return (threshold, label, vibration)
    return DISTANCE_TIERS[-1]


def build_message(label, direction, distance_label):
    if distance_label == "STOP":
        return f"STOP. {label} ahead."
    return f"{label} {direction}, {distance_label}"