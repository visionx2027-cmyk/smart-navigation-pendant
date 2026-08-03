import cv2
from ultralytics import YOLO
import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "camera_module"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
from capture import get_camera, get_frame
from voice import speak
model = YOLO("yolov8m.pt") 
cam = get_camera(0)

last_spoken = ""
last_speak_time = 6
cooldown = 5

while True:
    frame = get_frame(cam)
    results = model(frame, conf=0.6, verbose=False)
    annotated_frame = results[0].plot()
    cv2.imshow("VISIONX Object Detection", annotated_frame)

    boxes = results[0].boxes
    current_time = time.time()

    if len(boxes) > 0:
        top_box = boxes[boxes.conf.argmax()]
        class_id = int(top_box.cls[0])
        label = model.names[class_id]

        if label != last_spoken or (current_time - last_speak_time) > cooldown:
            message = f"{label} detected"
            print(f"Speaking: {message}")
            speak(message)
            last_spoken = label
            last_speak_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

