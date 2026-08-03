import time

def vibrate_simulated(pattern="single"):
    """
    Simulates vibration motor feedback.
    Replace with real GPIO code once the motor hardware is available.
    """
    if pattern == "single":
        print("[VIBRATION] Single short pulse (gentle alert)")
    elif pattern == "double":
        print("[VIBRATION] Double pulse (warning)")
    elif pattern == "continuous":
        print("[VIBRATION] Continuous buzz (danger!)")
    time.sleep(0.2)  # simulate motor response time