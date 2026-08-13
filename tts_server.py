import socket
import pyttsx3

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 5005))

print("Laptop TTS server listening on port 5005...")
while True:
    data, addr = sock.recvfrom(2048)
    text = data.decode()
    print(f"Speaking: {text}")
    engine = pyttsx3.init()      # fresh engine each time
    engine.setProperty('rate', 160)
    engine.say(text)
    engine.runAndWait()
    engine.stop()