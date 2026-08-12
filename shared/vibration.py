import RPi.GPIO as GPIO
import time

VIBRATION_PIN = 17  # GPIO17, physical pin 11
_gpio_initialized = False


def _init_gpio():
    global _gpio_initialized
    if _gpio_initialized:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(VIBRATION_PIN, GPIO.OUT)
    GPIO.output(VIBRATION_PIN, False)
    _gpio_initialized = True


def vibrate(pattern="off"):
    """
    Controls the real physical vibration motor via GPIO17.
    pattern: "off" | "double" | "continuous"
    - "off": no vibration (distance > 100cm)
    - "double": two short pulses (distance 30-100cm, warning zone)
    - "continuous": sustained vibration (distance <= 30cm, danger zone)
    """
    _init_gpio()

    if pattern == "off":
        GPIO.output(VIBRATION_PIN, False)

    elif pattern == "double":
        GPIO.output(VIBRATION_PIN, True)
        time.sleep(0.15)
        GPIO.output(VIBRATION_PIN, False)
        time.sleep(0.1)
        GPIO.output(VIBRATION_PIN, True)
        time.sleep(0.15)
        GPIO.output(VIBRATION_PIN, False)

    elif pattern == "continuous":
        GPIO.output(VIBRATION_PIN, True)
        time.sleep(0.3)
        # left ON — main loop will keep re-calling this each cycle while in danger zone


def stop_vibration():
    """Explicitly turns the motor off. Call this on exit/error."""
    GPIO.output(VIBRATION_PIN, False)


def cleanup():
    """Releases GPIO pin. Call this when the program shuts down."""
    stop_vibration()
    GPIO.cleanup(VIBRATION_PIN)