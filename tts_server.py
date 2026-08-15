import socket
import pyttsx3

HOST = "0.0.0.0"
PORT = 5005

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(5)

print(f"Laptop TTS server listening on TCP port {PORT}...")

while True:
    conn, addr = server.accept()

    try:
        data = conn.recv(2048)

        if data:
            text = data.decode("utf-8")
            print(f"Speaking: {text}")

            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            engine.say(text)
            engine.runAndWait()
            engine.stop()

    except Exception as e:
        print(f"TTS error: {e}")

    finally:
        conn.close()