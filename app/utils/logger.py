"""Centralized logging utility for the project.

Features:
- Colored console logs
- File logs (logs/app.log)
- Timestamped + module-aware output
"""

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name: str = "app", level=logging.INFO) -> logging.Logger:
    """Returns a configured logger instance.

    Safe to call from any module.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        return logger

    # Console handler
    console_handler = logging.StreamHandler()
    console_format = (
        "\033[36m[%(asctime)s] [%(name)s] \
        [%(levelname)s]\033[0m "
        "%(message)s"
    )
    console_handler.setFormatter(logging.Formatter(console_format, "%Y-%m-%d %H:%M:%S"))

    # File handler
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5_000_000,
        backupCount=3,
    )
    file_format = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    file_handler.setFormatter(logging.Formatter(file_format, "%Y-%m-%d %H:%M:%S"))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
