import random
import time

def read_distance_simulated():
    """
    Simulates an ultrasonic sensor reading.
    Replace this with real hardware code once ESP32 + HC-SR04 are available.
    """
    # Randomly simulate a distance between 10cm and 300cm
    distance_cm = random.uniform(10, 300)
    time.sleep(0.1)  # simulate real sensor read delay
    return round(distance_cm, 1)