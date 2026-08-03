import pyttsx3

def speak(text):
    """Speaks the given text out loud."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.say(text)
    engine.runAndWait()
    engine.stop()