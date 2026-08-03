"""
Centralized logging setup. Every module imports get_logger(__name__) instead
of using print() — this gives us timestamps, severity levels, and the option
to write logs to a file on the Pi for post-review debugging, without
touching any other module's code.
"""

import logging
import sys

_configured = False


def _configure_root_logger():
    global _configured
    if _configured:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("visionx.log", encoding="utf-8"),
        ],
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_root_logger()
    return logging.getLogger(name)