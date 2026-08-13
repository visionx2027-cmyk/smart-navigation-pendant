from picamera2 import Picamera2
import time


def get_camera(camera_index=0):
    """
    Opens the Pi Camera Module using picamera2.
    camera_index is kept as a parameter for compatibility with existing
    calls, but picamera2 handles device selection differently — this
    param is unused for now (single-camera setup).
    """
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    time.sleep(1)  # let the sensor settle before first capture
    return picam2


def get_frame(cam):
    """
    Reads a single frame from the Pi Camera. Returns a numpy array
    compatible with OpenCV (BGR-ish via RGB888 format), matching what
    the rest of the codebase (YOLO, face_recognition, OCR, cv2.imshow)
    already expects.
    """
    frame = cam.capture_array()
    if frame is None:
        raise RuntimeError("Failed to read frame from camera.")
    return frame


def save_frame(frame, filename="captured.jpg"):
    import cv2
    cv2.imwrite(filename, frame)
    print(f"Image saved as {filename}")