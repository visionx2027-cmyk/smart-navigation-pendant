import face_recognition
import pickle
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "camera_module"))
from capture import get_camera, get_frame, save_frame
import cv2

DATA_FILE = "known_faces/encodings.pkl"

def load_known_faces():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "names": []}

def save_known_faces(data):
    os.makedirs("known_faces", exist_ok=True)
    with open(DATA_FILE, "wb") as f:
        pickle.dump(data, f)

def register_new_face():
    name = input("Enter the person's name: ")

    cam = get_camera(0)
    print("Press 'c' to capture the photo for registration.")

    while True:
        frame = get_frame(cam)
        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            save_frame(frame, "temp_register.jpg")
            break

    cam.release()
    cv2.destroyAllWindows()

    image = face_recognition.load_image_file("temp_register.jpg")
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        print("No face detected in the photo. Try again with better lighting/angle.")
        return

    data = load_known_faces()
    data["encodings"].append(encodings[0])
    data["names"].append(name)
    save_known_faces(data)

    print(f"{name} registered successfully!")

if __name__ == "__main__":
    register_new_face()