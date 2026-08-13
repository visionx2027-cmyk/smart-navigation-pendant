"""Extracted, importable OCR — no camera, no loop, no keypress wait."""
import easyocr

reader = easyocr.Reader(['en'])


def read_text_from_frame(frame):
    results = reader.readtext(frame)
    text = " ".join([t for (_, t, c) in results if c > 0.2])
    return text if text else None