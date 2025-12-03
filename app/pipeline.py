"""Model pipeline loader for Stable Diffusion (HuggingFace Diffusers).

load_pipeline(...) returns a GPU-ready pipeline with memory optimizations.
"""

import os
from typing import Optional

import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
)
from dotenv import load_dotenv

from app.utils.logger import get_logger

logger = get_logger(__name__)
load_dotenv()


def _try_enable_xformers(pipe):
    try:
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xFormers memory-efficient attention.")
        else:
            logger.info("xFormers not available via API; skipping.")
    except Exception as err:
        logger.info(f"xFormers not enabled: {err}")


def load_pipeline(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    device: str = "cuda",
    use_fp16: bool = True,
    enable_xformers: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
    scheduler=None,
):
    """Load and return an optimized StableDiffusionPipeline."""
    if torch_dtype is None:
        torch_dtype = torch.float16 if use_fp16 and device == "cuda" else torch.float32

    if scheduler is None:
        try:
            scheduler = DPMSolverMultistepScheduler.from_pretrained(
                model_id,
                subfolder="scheduler",
            )
        except Exception:
            scheduler = None

    logger.info(f"Loading pipeline {model_id} " f"dtype={torch_dtype} on {device} ...")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
        scheduler=scheduler,
        use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
    )

    pipe = pipe.to(device)

    try:
        pipe.enable_attention_slicing()
        logger.info("Enabled attention slicing.")
    except Exception:
        logger.info("Attention slicing not available.")

    if enable_xformers:
        _try_enable_xformers(pipe)

    try:
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
            logger.info("Enabled VAE tiling.")
    except Exception:
        pass

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    logger.info("Pipeline loaded.")
    return pipe


def warmup_pipeline(
    pipe,
    prompt: str = "A photo of a cat",
    height: int = 512,
    width: int = 512,
):
    """Run a quick inference to allocate CUDA kernels and memory."""
    try:
        if hasattr(pipe, "parameters"):
            device = next(pipe.parameters()).device
        else:
            device = "cuda"

    except Exception:
        device = "cuda"

    try:
        gen = torch.Generator(device if device != "cpu" else "cpu").manual_seed(0)

        logger.info("Warmup: running one-step inference to initialize kernels.")

        _ = pipe(
            prompt=prompt,
            num_inference_steps=1,
            guidance_scale=1.0,
            height=height,
            width=width,
            generator=gen,
        )

        torch.cuda.empty_cache()
        logger.info("Warmup complete.")
    except Exception as err:
        logger.warning(f"Warmup failed: {err}")
