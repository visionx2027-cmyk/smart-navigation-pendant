import cv2
from capture import get_camera, get_frame, save_frame

cam = get_camera(0)

while True:
    frame = get_frame(cam)
    cv2.imshow("VISIONX Camera Preview", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        save_frame(frame)

cam.release()
cv2.destroyAllWindows()