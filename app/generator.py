"""Image generation wrapper around a loaded StableDiffusionPipeline.

Provides:
- generate_image(...) -> (PIL.Image, metadata)
- deterministic seed handling
"""

import time
from typing import Any, Dict, Optional

import torch

from app.utils.logger import get_logger

logger = get_logger(__name__)


def _validate_resolution(width: int, height: int):
    # clamp and snap to multiples of 64 (SD requirement)
    width = max(256, min(width, 768))
    height = max(256, min(height, 768))
    width = (width // 64) * 64
    height = (height // 64) * 64
    return int(width), int(height)


def generate_image(
    pipe,
    prompt: str,
    negative_prompt: Optional[str] = None,
    steps: int = 30,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: Optional[int] = None,
    device: str = "cuda",
):
    """Generate a single image and return (PIL.Image, metadata dict)."""
    start = time.time()
    width, height = _validate_resolution(width, height)

    # Generator for reproducibility
    if seed is None:
        # create a new seed and use it
        seed = int(torch.seed() & ((1 << 63) - 1))
    gen = torch.Generator(device if device != "cpu" else "cpu").manual_seed(int(seed))

    logger.info(
        (
            f"Generating: steps={steps}, cfg={guidance_scale},\
        res={width}x{height}, seed={seed}"
        )
    )

    # Use autocast for speed/precision management
    device_type = "cuda" if device != "cpu" else "cpu"
    with torch.autocast(device_type=device_type):
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            width=width,
            height=height,
            generator=gen,
        )

    img = result.images[0]  # PIL image
    elapsed = time.time() - start

    metadata: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
        "seed": int(seed),
        "elapsed_seconds": elapsed,
    }

    logger.info(f"Generation finished in {elapsed:.2f}s")
    return img, metadata
