"""Image-to-image generation using Stable Diffusion.

This module provides:
- prepare_img2img_pipeline: build an Img2Img pipeline from an existing txt2img pipe.
- generate_img2img: run image-to-image generation and return (PIL.Image, metadata).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

from app.utils.logger import get_logger

logger = get_logger(__name__)


def _validate_resolution(width: int, height: int) -> tuple[int, int]:
    """Clamp resolution to a safe range and snap to multiples of 64."""
    width = max(256, min(width, 768))
    height = max(256, min(height, 768))
    width = (width // 64) * 64
    height = (height // 64) * 64
    return int(width), int(height)


def _load_init_image(
    image: Union[Image.Image, str, Path],
    width: int,
    height: int,
) -> Image.Image:
    """Load and preprocess the init image for img2img."""
    if isinstance(image, (str, Path)):
        image = Image.open(image)

    if not isinstance(image, Image.Image):
        raise TypeError("init_image must be a PIL.Image or a valid image path.")

    image = image.convert("RGB")
    image = image.resize((width, height), resample=Image.LANCZOS)
    return image


def prepare_img2img_pipeline(
    base_pipe,
    model_id: str = "runwayml/stable-diffusion-v1-5",
) -> StableDiffusionImg2ImgPipeline:
    """Create an Img2Img pipeline that shares weights with the base txt2img pipe.

    Tries to use StableDiffusionImg2ImgPipeline.from_pipe to reuse:
    - UNet
    - VAE
    - text encoder
    - tokenizer
    - scheduler
    """
    try:
        img2img_pipe = StableDiffusionImg2ImgPipeline.from_pipe(base_pipe)
        logger.info("Created Img2Img pipeline from existing base pipeline.")
    except Exception as err:
        logger.info("from_pipe failed (%s); falling back to from_pretrained.", err)
        img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=base_pipe.unet.dtype,
            safety_checker=None,
        )
        device = next(base_pipe.unet.parameters()).device
        img2img_pipe = img2img_pipe.to(device)

    # memory optimizations similar to txt2img pipeline
    try:
        img2img_pipe.enable_attention_slicing()
        logger.info("Enabled attention slicing on Img2Img pipeline.")
    except Exception:
        logger.info("Attention slicing not available on Img2Img pipeline.")

    try:
        if hasattr(img2img_pipe.vae, "enable_tiling"):
            img2img_pipe.vae.enable_tiling()
            logger.info("Enabled VAE tiling on Img2Img pipeline.")
    except Exception:
        pass

    return img2img_pipe


def generate_img2img(
    pipe: StableDiffusionImg2ImgPipeline,
    init_image: Union[Image.Image, str, Path],
    prompt: str,
    negative_prompt: Optional[str] = None,
    strength: float = 0.7,
    steps: int = 30,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: Optional[int] = None,
    device: str = "cuda",
) -> tuple[Image.Image, Dict[str, Any]]:
    """Run image-to-image generation.

    Args:
        pipe: A StableDiffusionImg2ImgPipeline.
        init_image: Base image (PIL or path).
        prompt: Text prompt to guide the transformation.
        negative_prompt: What to avoid in the output.
        strength: How strong the transformation is (0-1).
        steps: Number of inference steps.
        guidance_scale: Prompt adherence strength.
        width: Target width (snapped to 64 multiple).
        height: Target height (snapped to 64 multiple).
        seed: Optional random seed for reproducibility.
        device: "cuda" or "cpu".

    Returns:
        (PIL.Image, metadata dict)
    """
    if not (0.0 < strength <= 1.0):
        raise ValueError("strength must be in (0, 1].")

    start = time.time()
    width, height = _validate_resolution(width, height)
    init_image = _load_init_image(init_image, width, height)

    # Seed handling
    if seed is None:
        seed = int(torch.seed() & ((1 << 63) - 1))

    gen = torch.Generator(device if device != "cpu" else "cpu").manual_seed(int(seed))

    logger.info(
        "Img2Img: steps=%s cfg=%s strength=%.2f res=%sx%s seed=%s",
        steps,
        guidance_scale,
        strength,
        width,
        height,
        seed,
    )

    device_type = "cuda" if device != "cpu" else "cpu"
    with torch.autocast(device_type=device_type):
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            image=init_image,
            strength=float(strength),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=gen,
        )

    out_image = result.images[0]
    elapsed = time.time() - start

    metadata: Dict[str, Any] = {
        "mode": "img2img",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
        "seed": int(seed),
        "strength": float(strength),
        "elapsed_seconds": elapsed,
    }

    logger.info("Img2Img finished in %.2fs", elapsed)
    return out_image, metadata
