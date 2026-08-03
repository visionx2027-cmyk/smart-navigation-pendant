import random
import time

def read_gps_simulated():
    """
    Simulates GPS coordinates.
    Replace with real hardware reading once the GPS module is available.
    Using a rough Bengaluru-area coordinate range for realistic testing.
    """
    latitude = round(random.uniform(12.90, 13.05), 6)
    longitude = round(random.uniform(77.55, 77.70), 6)
    time.sleep(1)  # real GPS modules update roughly once per second
    return latitude, longitude