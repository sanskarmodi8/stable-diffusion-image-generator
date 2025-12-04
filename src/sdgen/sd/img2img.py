"""Img2Img pipeline setup and generation utilities."""

from __future__ import annotations

import time

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

from src.sdgen.sd.models import GenerationMetadata, Img2ImgConfig
from src.sdgen.utils.common import validate_resolution
from src.sdgen.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_img2img_pipeline(
    base_pipe: StableDiffusionImg2ImgPipeline,
    model_id: str = "runwayml/stable-diffusion-v1-5",
) -> StableDiffusionImg2ImgPipeline:
    """Create an Img2Img pipeline using an existing base pipeline.

    Attempts `from_pipe` first for efficiency, then falls back to
    a clean `from_pretrained` load if necessary.

    Args:
        base_pipe: Loaded text-to-image Stable Diffusion pipeline.
        model_id: Fallback Hugging Face model ID.

    Returns:
        Configured `StableDiffusionImg2ImgPipeline`.
    """
    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pipe(base_pipe)
        logger.info("Img2Img pipeline created via from_pipe().")
    except Exception as exc:
        logger.warning("from_pipe() failed: %s → falling back.", exc)
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=base_pipe.unet.dtype,
            safety_checker=None,
        )
        device = next(base_pipe.unet.parameters()).device
        pipe = pipe.to(device)

    # Optimizations
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    try:
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
    except Exception:
        pass

    return pipe


def generate_img2img(
    pipe: StableDiffusionImg2ImgPipeline,
    cfg: Img2ImgConfig,
    init_image: Image.Image,
) -> tuple[Image.Image, GenerationMetadata]:
    """Run Img2Img generation using the configured pipeline and metadata config.

    Args:
        pipe: Stable Diffusion Img2Img pipeline.
        cfg: Img2Img inference settings (prompt, steps, etc.).
        init_image: The source image to transform.

    Raises:
        ValueError: If strength is outside (0, 1].

    Returns:
        A tuple of `(output_image, metadata)`.
    """
    if not (0.0 < cfg.strength <= 1.0):
        raise ValueError("strength must be in (0, 1].")

    width, height = validate_resolution(cfg.width, cfg.height)
    start = time.time()

    # Deterministic seed
    seed = cfg.seed
    if seed is None:
        seed = int(torch.seed() & ((1 << 63) - 1))

    # Resize input
    init = init_image.convert("RGB").resize((width, height), Image.LANCZOS)

    # Correct generator device
    device = cfg.device if cfg.device in ("cuda", "cpu") else "cuda"
    generator = torch.Generator(device).manual_seed(int(seed))

    logger.info(
        "img2img: steps=%s cfg=%s strength=%.2f res=%sx%s seed=%s",
        cfg.steps,
        cfg.guidance_scale,
        cfg.strength,
        width,
        height,
        seed,
    )

    # Autocast context
    autocast_device = "cuda" if device == "cuda" else "cpu"
    with torch.autocast(device_type=autocast_device):
        out = pipe(
            prompt=cfg.prompt,
            negative_prompt=cfg.negative_prompt or None,
            image=init,
            strength=float(cfg.strength),
            num_inference_steps=int(cfg.steps),
            guidance_scale=float(cfg.guidance_scale),
            generator=generator,
        )

    img = out.images[0]
    elapsed = time.time() - start

    meta = GenerationMetadata(
        mode="img2img",
        prompt=cfg.prompt,
        negative_prompt=cfg.negative_prompt or "",
        steps=int(cfg.steps),
        guidance_scale=float(cfg.guidance_scale),
        width=width,
        height=height,
        seed=int(seed),
        strength=float(cfg.strength),
        elapsed_seconds=float(elapsed),
    )
    return img, meta
