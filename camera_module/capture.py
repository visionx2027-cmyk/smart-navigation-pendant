import cv2

def get_camera(camera_index=0):
    """Opens the camera and returns the camera object."""
    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        raise RuntimeError("Could not open camera. Check the camera index or connection.")
    return cam

def get_frame(cam):
    """Reads a single frame from the camera."""
    success, frame = cam.read()
    if not success:
        raise RuntimeError("Failed to read frame from camera.")
    return frame

def save_frame(frame, filename="captured.jpg"):
    """Saves a frame as an image file."""
    cv2.imwrite(filename, frame)
    print(f"Image saved as {filename}")