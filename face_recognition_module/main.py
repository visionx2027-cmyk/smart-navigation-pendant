import face_recognition
import cv2
import pickle
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "camera_module"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
from capture import get_camera, get_frame
from voice import speak

DATA_FILE = "known_faces/encodings.pkl"

def load_known_faces():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "names": []}

known_data = load_known_faces()
known_encodings = known_data["encodings"]
known_names = known_data["names"]

cam = get_camera(0)
last_spoken = ""
last_speak_time = 0
cooldown = 5

while True:
    frame = get_frame(cam)

    # face_recognition expects RGB, OpenCV gives BGR — convert
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

        # Draw box + label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        current_time = time.time()
        if name != last_spoken or (current_time - last_speak_time) > cooldown:
            speak(f"{name} detected" if name != "Unknown" else "Unknown person detected")
            last_spoken = name
            last_speak_time = current_time

    cv2.imshow("VISIONX Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()