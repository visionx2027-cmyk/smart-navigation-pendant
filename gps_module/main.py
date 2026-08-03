import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
from voice import speak
from gps_reader import read_gps_simulated

print("Reading simulated GPS coordinates. Press Ctrl+C to stop.")

try:
    while True:
        lat, lon = read_gps_simulated()
        print(f"Latitude: {lat}, Longitude: {lon}")
        time.sleep(2)  # print every 2 seconds, don't spam terminal

except KeyboardInterrupt:
    print("\nStopped.")