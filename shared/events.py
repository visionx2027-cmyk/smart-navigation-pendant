"""
Defines the Event object — the common "message format" every module uses
to report something worth announcing to the decision engine.

Why this exists: object_detection, obstacle_detection, ocr, face_recognition,
and gps all detect completely different things. Without a shared format,
the decision engine would need custom handling for each module's output.
With this, every module just produces an Event, and the decision engine
only ever deals with one type of object.
"""

from dataclasses import dataclass, field
import time


@dataclass
class Event:
    source: str          # which module raised this, e.g. "object_detection"
    priority: int         # from shared.constants (lower = more urgent)
    message: str          # the actual text to speak
    tag: str              # identifies *what* this event is about, e.g. "person_left_close"
                          # used to detect duplicates — same tag = same situation
    timestamp: float = field(default_factory=time.time)

    def is_duplicate_of(self, other: "Event") -> bool:
        """Two events are duplicates if they're the same tag from the same source."""
        return self.source == other.source and self.tag == other.tag