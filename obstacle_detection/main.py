import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
from voice import speak
from vibration import vibrate, cleanup
from sensor import read_distance_real

DANGER_ZONE = 30
WARNING_ZONE = 100

last_zone = None
last_speak_time = 0
cooldown = 3

print("Reading REAL obstacle distance from HC-SR04. Press Ctrl+C to stop.")

try:
    while True:
        distance = read_distance_real()

        if distance is None:
            print("No echo received (out of range or wiring issue)")
            vibrate("off")
            time.sleep(0.3)
            continue

        print(f"Distance: {distance} cm")
        current_time = time.time()

        if distance < DANGER_ZONE:
            zone = "danger"
            message = "Obstacle very close, stop"
            vibration_pattern = "continuous"
        elif distance < WARNING_ZONE:
            zone = "warning"
            message = "Obstacle ahead, be careful"
            vibration_pattern = "double"
        else:
            zone = "clear"
            message = None
            vibration_pattern = "off"

        # Vibration runs every cycle (reflects CURRENT distance, not gated by cooldown)
        vibrate(vibration_pattern)

        # Voice still respects cooldown/zone-change logic, unchanged from before
        if zone != last_zone or (current_time - last_speak_time) > cooldown:
            if message:
                print(f"Speaking: {message}")
                speak(message)
            last_zone = zone
            last_speak_time = current_time

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    cleanup()