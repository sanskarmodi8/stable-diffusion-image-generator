"""Configuration exports for the sdgen package.

This module re-exports commonly used configuration paths and settings
so they can be imported directly from `src.sdgen.config`.
"""

from __future__ import annotations

from .paths import (
    ASSETS_ROOT,
    HISTORY_ENTRIES_DIR,
    HISTORY_FULL_DIR,
    HISTORY_ROOT,
    HISTORY_THUMBS_DIR,
    LOGS_ROOT,
    PROJECT_ROOT,
)
from .settings import AppSettings

__all__ = [
    "AppSettings",
    "PROJECT_ROOT",
    "ASSETS_ROOT",
    "HISTORY_ROOT",
    "HISTORY_ENTRIES_DIR",
    "HISTORY_THUMBS_DIR",
    "HISTORY_FULL_DIR",
    "LOGS_ROOT",
]
