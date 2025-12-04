"""Unified interface for image upscaling.

This module selects an upscaling backend at runtime.
Currently supported:
- NCNN RealESRGAN (recommended)

Planned:
- Stable Diffusion-based upscaler
"""

from __future__ import annotations

from typing import Optional

from PIL import Image

from src.sdgen.upscaler.realesrgan import NCNNUpscaler
from src.sdgen.utils.logger import get_logger

logger = get_logger(__name__)


class Upscaler:
    """Unified high-level upscaler wrapper.

    Args:
        scale: Target scale factor. Typically 2 or 4.
        prefer: Preferred backend name:
            - "ncnn": NCNN RealESRGAN (local, fast)
            - "auto": Try known engines in order

    Raises:
        RuntimeError: If no backend could be initialized.
        ValueError: Invalid scale value given.
    """

    _VALID_SCALES = {2, 4}
    _BACKENDS_ORDER = ("ncnn",)

    def __init__(self, scale: float = 2.0, prefer: str = "ncnn") -> None:
        """Initialize upscaler class."""
        if int(scale) not in self._VALID_SCALES:
            msg = "Scale must be 2 or 4 for RealESRGAN. Got: %s"
            raise ValueError(msg % scale)

        self.scale = int(scale)
        self.engine: Optional[object] = None

        logger.info("Upscaler init (prefer=%s, scale=%s)", prefer, self.scale)

        if prefer == "auto":
            self._init_auto()
        elif prefer == "ncnn":
            self._init_ncnn()
        else:
            msg = "Unknown upscaler backend: %s"
            raise ValueError(msg % prefer)

        if self.engine is None:
            raise RuntimeError("No valid upscaler engine available.")

    def _init_auto(self) -> None:
        """Try available engines in priority order."""
        for backend in self._BACKENDS_ORDER:
            try:
                if backend == "ncnn":
                    self._init_ncnn()
                    return
            except Exception as err:  # noqa: BLE001
                logger.warning("Upscaler init failed (%s): %s", backend, err)

    def _init_ncnn(self) -> None:
        """Initialize RealESRGAN NCNN backend."""
        try:
            self.engine = NCNNUpscaler(scale=self.scale)
            logger.info("Using NCNN RealESRGAN engine.")
        except Exception as err:  # noqa: BLE001
            logger.warning("NCNN RealESRGAN init failed: %s", err)
            self.engine = None

    def upscale(self, image: Image.Image) -> Image.Image:
        """Upscale the given image.

        Args:
            image: Input PIL image.

        Returns:
            The upscaled PIL image.

        Raises:
            RuntimeError: If the engine is not initialized.
        """
        if self.engine is None:
            raise RuntimeError("Upscaler is not initialized.")
        return self.engine.upscale(image)
