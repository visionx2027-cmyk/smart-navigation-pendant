import lgpio
import time

TRIG_PIN = 23
ECHO_PIN = 24

chip = lgpio.gpiochip_open(0)

lgpio.gpio_claim_output(chip, TRIG_PIN, 0)
lgpio.gpio_claim_input(chip, ECHO_PIN)


def read_distance_real(timeout=0.04):
    # Send 10 microsecond trigger pulse
    lgpio.gpio_write(chip, TRIG_PIN, 1)
    time.sleep(0.00001)
    lgpio.gpio_write(chip, TRIG_PIN, 0)

    start_wait = time.time()

    # Wait for echo to become HIGH
    while lgpio.gpio_read(chip, ECHO_PIN) == 0:
        if time.time() - start_wait > timeout:
            return None

    start_time = time.time()

    # Wait for echo to become LOW
    while lgpio.gpio_read(chip, ECHO_PIN) == 1:
        if time.time() - start_time > timeout:
            return None

    elapsed = time.time() - start_time

    distance_cm = (elapsed * 34300) / 2

    return round(distance_cm, 1)


def cleanup():
    lgpio.gpio_free(chip, TRIG_PIN)
    lgpio.gpio_free(chip, ECHO_PIN)
    lgpio.gpiochip_close(chip)