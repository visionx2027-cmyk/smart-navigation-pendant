import cv2
import easyocr
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "camera_module"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
from capture import get_camera, get_frame
from voice import speak

# Load the OCR reader once (English only, for now)
reader = easyocr.Reader(['en'])

cam = get_camera(0)
last_text = ""

print("Press 'r' to read text, 'q' to quit.")

while True:
    frame = get_frame(cam)
    cv2.imshow("VISIONX OCR", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        print("Reading text... (this may take a few seconds, please wait)")
        results = reader.readtext(frame)

        print(f"Raw results: {results}")  # shows everything detected, before filtering

        detected_text = " ".join([text for (_, text, conf) in results if conf > 0.2])

        if detected_text:
            print(f"Detected: {detected_text}")
            speak(detected_text)
            last_text = detected_text
        else:
            print("No readable text found.")
            speak("No text found")

    elif key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()