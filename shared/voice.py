import socket

LAPTOP_IP = "192.168.0.151"   # <-- your laptop's real IP
LAPTOP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def speak(text):
    """Sends text to laptop over WiFi to be spoken aloud."""
    try:
        sock.sendto(text.encode(), (LAPTOP_IP, LAPTOP_PORT))
    except Exception as e:
        print(f"Failed to send audio text: {e}")