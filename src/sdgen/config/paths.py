"""Path configuration for sdgen.

All filesystem paths are resolved relative to the project root.
The project root is detected by walking upward until a marker
file (e.g., `pyproject.toml` or `.git`) is found.
"""

from __future__ import annotations

from pathlib import Path


def _detect_project_root() -> Path:
    """Return the project root by scanning upward for a marker file."""
    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent

    # Fallback: use the last resolved parent
    return current.parents[-1]


PROJECT_ROOT: Path = _detect_project_root()

ASSETS_ROOT: Path = PROJECT_ROOT / "src" / "assets"
ASSETS_ROOT.mkdir(parents=True, exist_ok=True)

HISTORY_ROOT: Path = ASSETS_ROOT / "history"
HISTORY_ENTRIES_DIR: Path = HISTORY_ROOT / "entries"
HISTORY_THUMBS_DIR: Path = HISTORY_ROOT / "thumbnails"
HISTORY_FULL_DIR: Path = HISTORY_ROOT / "full"

for p in [
    HISTORY_ROOT,
    HISTORY_ENTRIES_DIR,
    HISTORY_THUMBS_DIR,
    HISTORY_FULL_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)

LOGS_ROOT: Path = PROJECT_ROOT / "logs"
LOGS_ROOT.mkdir(parents=True, exist_ok=True)
