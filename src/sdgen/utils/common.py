"""Utility helpers for image conversion, resolution validation, and formatting."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from PIL import Image


def validate_resolution(width: int, height: int) -> tuple[int, int]:
    """Clamp and align the resolution to multiples of 64 within the SD range.

    Stable Diffusion models expect spatial dimensions that are multiples of 64.
    The allowed range is clamped to [256, 768] to avoid excessive memory use.

    Args:
        width: Requested width in pixels.
        height: Requested height in pixels.

    Returns:
        A (width, height) tuple aligned to the valid grid.
    """
    width = (max(256, min(width, 768)) // 64) * 64
    height = (max(256, min(height, 768)) // 64) * 64
    return width, height


def to_pil(image: Any) -> Image.Image:
    """Convert a numpy array to a PIL image, or return the existing PIL image.

    Supports:
    - uint8 arrays in shape (H, W) or (H, W, C)
    - float arrays assumed to be normalized in [0, 1]
    - PIL.Image is returned unchanged

    Args:
        image: Input image data, either PIL.Image or numpy.ndarray.

    Returns:
        A PIL.Image instance.

    Raises:
        TypeError: If the input type is unsupported.
    """
    if isinstance(image, Image.Image):
        return image

    if isinstance(image, np.ndarray):
        arr = image

        # Normalize floats to uint8 safely
        if np.issubdtype(arr.dtype, np.floating):
            # Clip first to avoid wraparound
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype("uint8")
        elif arr.dtype != np.uint8:
            arr = arr.astype("uint8")

        # Grayscale → RGB
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)

        # Drop alpha channel if present
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]

        return Image.fromarray(arr)

    raise TypeError(
        f"Expected PIL.Image or numpy.ndarray for 'image', got {type(image).__name__!r}"
    )


def pretty_json(data: Any) -> str:
    """Return a pretty-printed JSON string representation of data.

    Args:
        data: Any JSON-serializable object.

    Returns:
        A formatted JSON string. If serialization fails, a best-effort string
        representation is returned.
    """
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


def short_prompt(text: str | None, max_len: int = 50) -> str:
    """Return a compact single-line prompt suitable for labels.

    Removes newlines and truncates with an ellipsis if longer than max_len.

    Args:
        text: The full text prompt.
        max_len: Maximum number of characters including ellipsis.

    Returns:
        A short display string.
    """
    if not text:
        return ""

    text = text.replace("\n", " ")
    if len(text) <= max_len:
        return text

    # Reserve 1 char for ellipsis
    return text[: max_len - 1] + "…"
