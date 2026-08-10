import random
import time

# Holds the most recent distance reading so other modules can read it
# without triggering a new reading themselves.
_latest_distance_cm = 300  # safe default


def read_distance_simulated():
    """
    Simulates an ultrasonic sensor reading.
    NOT currently used by object_detection (camera-based estimate is used
    instead until real hardware is connected) — kept here so the standalone
    obstacle_detection module still works independently if you run it on
    its own.
    """
    global _latest_distance_cm
    distance_cm = random.uniform(10, 300)
    time.sleep(0.1)
    _latest_distance_cm = round(distance_cm, 1)
    return _latest_distance_cm


def get_latest_distance_m():
    """Returns the last known distance in meters."""
    return _latest_distance_cm / 100


def read_distance_real():
    """
    PLACEHOLDER for Monday — real HC-SR04 reading via GPIO.
    Once wired up, this replaces the camera-based estimate in
    object_detection/main.py. Fill in with the GPIO TRIG/ECHO code
    once hardware is confirmed working.
    """
    raise NotImplementedError("Wire this up once HC-SR04 is connected on Monday")