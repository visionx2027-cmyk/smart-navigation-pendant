"""
Custom exceptions for VISIONX. Using specific exception types (instead of
generic Exception) lets main.py catch and handle each failure mode
differently — e.g. a camera disconnect should maybe retry, but a GPS
timeout should just skip that reading and continue.
"""


class VisionXError(Exception):
    """Base class for all VISIONX-specific errors."""
    pass


class CameraError(VisionXError):
    """Raised when the camera fails to open or read a frame."""
    pass


class SensorError(VisionXError):
    """Raised when a hardware sensor (ultrasonic, GPS) fails to respond."""
    pass


class OCRError(VisionXError):
    """Raised when OCR fails to process an image."""
    pass


class FaceRecognitionError(VisionXError):
    """Raised when face recognition setup or matching fails."""
    pass