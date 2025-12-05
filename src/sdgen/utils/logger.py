"""Lightweight logger factory for the SDGen application.

This module centralizes logger configuration to ensure consistent formatting,
file rotation, and prevention of duplicate handlers during repeated imports.
"""

from __future__ import annotations

import logging
from logging import Handler, Logger
from logging.handlers import RotatingFileHandler

from sdgen.config import LOGS_ROOT

# Ensure logs directory exists
LOGS_ROOT.mkdir(parents=True, exist_ok=True)

# Cache prevents repeated handler installation for the same logger name
_LOGGER_CACHE: dict[str, Logger] = {}


def _build_handler() -> Handler:
    """Return a rotating file handler with unified log formatting.

    The handler writes to `app.log` under LOGS_ROOT and uses log rotation
    to cap file size and maintain up to 3 backups.
    """
    log_file = LOGS_ROOT / "app.log"
    handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=5_000_000,  # ~5 MB
        backupCount=3,
    )
    fmt = "%(asctime)s [%(name)s] [%(levelname)s] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    return handler


def get_logger(name: str) -> Logger:
    """Return a configured logger with rotating file and console handlers.

    The returned logger:
    - uses INFO level by default
    - writes to both stdout and a rotating log file
    - does not propagate to root logger
    - never duplicates handlers for the same name

    Args:
        name: Distinct logger name, generally the module name.

    Returns:
        A configured `logging.Logger` instance.
    """
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Guard against accidentally adding handlers multiple times
    if not logger.handlers:
        logger.addHandler(_build_handler())

        stream = logging.StreamHandler()
        stream.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s]" + "[%(levelname)s] %(message)s")
        )
        logger.addHandler(stream)

    logger.propagate = False
    _LOGGER_CACHE[name] = logger
    return logger
