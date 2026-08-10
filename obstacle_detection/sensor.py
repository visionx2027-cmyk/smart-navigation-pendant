import RPi.GPIO as GPIO
import time

TRIG_PIN = 23   # GPIO23, physical pin 16
ECHO_PIN = 24   # GPIO24, physical pin 18 — change if wired differently

_latest_distance_cm = 300
_gpio_initialized = False


def _init_gpio():
    global _gpio_initialized
    if _gpio_initialized:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)
    GPIO.output(TRIG_PIN, False)
    time.sleep(0.5)
    _gpio_initialized = True


def read_distance_real(timeout=0.04):
    global _latest_distance_cm
    _init_gpio()

    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)

    start_time = time.time()
    stop_time = time.time()
    timeout_start = time.time()

    while GPIO.input(ECHO_PIN) == 0:
        start_time = time.time()
        if start_time - timeout_start > timeout:
            return None

    timeout_start = time.time()
    while GPIO.input(ECHO_PIN) == 1:
        stop_time = time.time()
        if stop_time - timeout_start > timeout:
            return None

    elapsed = stop_time - start_time
    distance_cm = (elapsed * 34300) / 2
    _latest_distance_cm = round(distance_cm, 1)
    return _latest_distance_cm


def get_latest_distance_m():
    return _latest_distance_cm / 100


def cleanup():
    GPIO.cleanup()