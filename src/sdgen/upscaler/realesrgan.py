"""NCNN RealESRGAN upscaler wrapper.

This module exposes:
    - NCNNUpscaler: lightweight RealESRGAN upscaling (2× or 4×)
      backed by realesrgan-ncnn-py.
"""

from __future__ import annotations

from typing import Final

from PIL import Image
from realesrgan_ncnn_py import Realesrgan

from sdgen.utils.logger import get_logger

logger = get_logger(__name__)

# Map scale → realesrgan-ncnn model index
_SCALE_MODEL_MAP: Final[dict[int, int]] = {
    2: 3,  # realesrgan-x2plus
    4: 0,  # realesrgan-x4plus
}


class NCNNUpscaler:
    """NCNN RealESRGAN engine using realesrgan-ncnn-py.

    This class provides 2× or 4× super-resolution on CPU/GPU
    without requiring the full PyTorch RealESRGAN stack.

    Args:
        scale: Target scale factor. Valid values: 2 or 4.

    Raises:
        ValueError: If an unsupported scale is provided.
        RuntimeError: If the model cannot be loaded.
    """

    def __init__(self, scale: int = 2) -> None:
        """Initialize realesrgan."""
        if scale not in _SCALE_MODEL_MAP:
            msg = "Scale must be 2 or 4 for NCNN RealESRGAN, got: %s"
            raise ValueError(msg % scale)

        self.scale: int = scale
        model_index = _SCALE_MODEL_MAP[scale]

        logger.info(
            "Initializing NCNN RealESRGAN (scale=%s, model_index=%s)",
            scale,
            model_index,
        )

        try:
            self.model = Realesrgan(model=model_index)
        except Exception as exc:  # noqa: BLE001
            msg = "Failed to initialize Realesrgan engine: %s"
            logger.error(msg, exc)
            raise RuntimeError(msg % exc) from exc

    def upscale(self, image: Image.Image) -> Image.Image:
        """Upscale a PIL image using the NCNN RealESRGAN engine.

        Args:
            image: A PIL.Image instance.

        Returns:
            The upscaled PIL.Image.

        Raises:
            TypeError: If the input is not a PIL.Image.
        """
        if not isinstance(image, Image.Image):
            msg = "Input must be a PIL.Image, got: %s"
            raise TypeError(msg % type(image).__name__)

        logger.info(
            "Upscaling image (%sx%s) by %sx",
            image.width,
            image.height,
            self.scale,
        )

        return self.model.process_pil(image)
