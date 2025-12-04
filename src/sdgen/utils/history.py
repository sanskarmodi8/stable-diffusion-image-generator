"""History storage and indexing utilities for generated images.

This module handles:
- Writing a GenerationMetadata entry (JSON + images)
- Maintaining a compact index.json for fast history listing
- Atomic writes to avoid corruption on crash
- Optional deletion of individual history entries
"""

from __future__ import annotations

import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from src.sdgen.config import (
    HISTORY_ENTRIES_DIR,
    HISTORY_FULL_DIR,
    HISTORY_ROOT,
    HISTORY_THUMBS_DIR,
)
from src.sdgen.sd.models import GenerationMetadata, HistorySummary
from src.sdgen.utils.logger import get_logger

logger = get_logger(__name__)

# Ensure directories exist early
for _path in (
    HISTORY_ROOT,
    HISTORY_ENTRIES_DIR,
    HISTORY_THUMBS_DIR,
    HISTORY_FULL_DIR,
):
    _path.mkdir(parents=True, exist_ok=True)

INDEX_FILE = HISTORY_ROOT / "index.json"


# Internal helpers
def _atomic_write(path: Path, data: bytes) -> None:
    """Write bytes atomically to avoid partial writes on crash."""
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _read_index() -> List[Dict[str, Any]]:
    """Return list of summary dicts from index.json."""
    if not INDEX_FILE.exists():
        return []
    try:
        with INDEX_FILE.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read history index: %s", exc)
        return []


def _write_index(index: List[Dict[str, Any]]) -> None:
    """Persist index.json safely."""
    try:
        payload = json.dumps(
            index,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")
        _atomic_write(INDEX_FILE, payload)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to write history index: %s", exc)


def _save_images(
    entry_id: str,
    image: Image.Image,
    thumb_max_size: int = 256,
) -> Tuple[str, str]:
    """Save full PNG and resized thumbnail for given entry ID."""
    full_path = HISTORY_FULL_DIR / f"{entry_id}.png"
    thumb_path = HISTORY_THUMBS_DIR / f"{entry_id}.png"

    image.save(full_path, format="PNG")

    thumb = image.copy()
    thumb.thumbnail((thumb_max_size, thumb_max_size), Image.LANCZOS)
    thumb.save(thumb_path, format="PNG")

    return str(full_path), str(thumb_path)


# Public API
def save_history_entry(
    metadata: GenerationMetadata,
    image: Image.Image,
) -> GenerationMetadata:
    """Write a new history entry: images, metadata, and update index.json.

    Args:
        metadata: Populated GenerationMetadata (without paths or id)
        image: PIL image to save

    Returns:
        The metadata object, updated with id, timestamp, and image paths.
    """
    entry_id = metadata.id or str(uuid.uuid4())
    full_path, thumb_path = _save_images(entry_id, image)

    # Update metadata object
    metadata.id = entry_id
    metadata.full_image = full_path
    metadata.thumbnail = thumb_path
    if not metadata.timestamp:
        metadata.timestamp = datetime.utcnow().isoformat()

    # Write metadata JSON
    entry_file = HISTORY_ENTRIES_DIR / f"{entry_id}.json"
    try:
        payload = json.dumps(
            metadata.to_dict(),
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")
        _atomic_write(entry_file, payload)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to write metadata file: %s", exc)

    # Insert at top of index
    try:
        index = _read_index()
        summary = HistorySummary(
            id=entry_id,
            prompt=metadata.prompt,
            mode=metadata.mode,
            seed=metadata.seed,
            width=metadata.width,
            height=metadata.height,
            timestamp=metadata.timestamp,
            thumbnail=thumb_path,
        )
        # de-dupe old
        index = [summary.to_dict()] + [e for e in index if e.get("id") != entry_id]
        # cap history length
        index = index[:500]
        _write_index(index)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to update history index: %s", exc)

    logger.info("Saved history entry %s", entry_id)
    return metadata


def list_history(n: int = 50) -> List[Dict[str, Any]]:
    """Return newest history summary dicts, up to n."""
    index = _read_index()
    return index[:n]


def load_entry(entry_id: str) -> Optional[Dict[str, Any]]:
    """Return the full metadata dict for a specific entry_id, or None."""
    path = HISTORY_ENTRIES_DIR / f"{entry_id}.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load entry %s: %s", entry_id, exc)
        return None


def delete_history_entry(entry_id: str) -> bool:
    """Delete a history entry JSON + images and update index.json.

    Args:
        entry_id: History entry ID to delete.

    Returns:
        True if an entry was removed, False if not found.
    """
    index = _read_index()
    new_index: List[Dict[str, Any]] = []
    removed = False

    for item in index:
        if item.get("id") != entry_id:
            new_index.append(item)
            continue

        removed = True
        # Delete thumbnail
        thumb = item.get("thumbnail")
        if thumb:
            thumb_path = Path(thumb)
            if thumb_path.exists():
                try:
                    thumb_path.unlink()
                except Exception:  # noqa: BLE001
                    pass

        # Delete full image (only known from metadata)
        entry = load_entry(entry_id)
        if entry:
            full = entry.get("full_image")
            if full:
                full_path = Path(full)
                if full_path.exists():
                    try:
                        full_path.unlink()
                    except Exception:  # noqa: BLE001
                        pass

        # Delete entry file
        json_path = HISTORY_ENTRIES_DIR / f"{entry_id}.json"
        if json_path.exists():
            try:
                json_path.unlink()
            except Exception:  # noqa: BLE001
                pass

    if not removed:
        return False

    _write_index(new_index)
    logger.info("Deleted history entry %s", entry_id)
    return True
