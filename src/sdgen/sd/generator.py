"""Text-to-image generation with clean metadata output."""

from __future__ import annotations

import time
from typing import Tuple

import torch
from PIL import Image

from src.sdgen.sd.models import GenerationMetadata, Txt2ImgConfig
from src.sdgen.utils.common import validate_resolution
from src.sdgen.utils.logger import get_logger

logger = get_logger(__name__)


def generate_image(
    pipe: any,
    cfg: Txt2ImgConfig,
) -> Tuple[Image.Image, GenerationMetadata]:
    """Generate an image from text using a Stable Diffusion pipeline.

    Args:
        pipe: A diffusers StableDiffusionPipeline instance.
        cfg: Structured configuration for text-to-image generation.

    Returns:
        A tuple of (PIL image, GenerationMetadata).
    """
    width, height = validate_resolution(cfg.width, cfg.height)
    start = time.time()

    seed = cfg.seed
    if seed is None:
        seed = int(torch.seed() & ((1 << 63) - 1))

    device = cfg.device
    gen = torch.Generator("cpu" if device == "cpu" else device).manual_seed(int(seed))

    logger.info(
        "txt2img: steps=%s cfg=%s res=%sx%s seed=%s",
        cfg.steps,
        cfg.guidance_scale,
        width,
        height,
        seed,
    )

    autocast_device = device if device == "cuda" else "cpu"
    with torch.autocast(device_type=autocast_device):
        out = pipe(
            prompt=cfg.prompt,
            negative_prompt=cfg.negative_prompt or None,
            width=width,
            height=height,
            num_inference_steps=int(cfg.steps),
            guidance_scale=float(cfg.guidance_scale),
            generator=gen,
        )

    img = out.images[0]
    elapsed = time.time() - start

    meta = GenerationMetadata(
        mode="txt2img",
        prompt=cfg.prompt,
        negative_prompt=cfg.negative_prompt or "",
        steps=int(cfg.steps),
        guidance_scale=float(cfg.guidance_scale),
        width=width,
        height=height,
        seed=int(seed),
        elapsed_seconds=float(elapsed),
    )
    return img, meta
