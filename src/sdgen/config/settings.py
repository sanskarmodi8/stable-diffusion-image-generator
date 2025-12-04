"""Application runtime settings for sdgen.

AppSettings reads configuration values from environment variables at
process start and exposes them as strongly typed attributes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class AppSettings:
    """Config values for the Stable Diffusion app.

    Supported environment variables:
    - MODEL_ID: HuggingFace model name
    - XFORMERS: 1/0 to enable xformers
    - WARMUP: 1/0 to warm up CUDA kernels
    - PORT: server port for Gradio
    - HOST: server host address
    - SHARE: enable Gradio public sharing link
    """

    model_id1: str = os.getenv("MODEL_ID1", "runwayml/stable-diffusion-v1-5")
    model_id2: str = os.getenv("MODEL_ID2", "stabilityai/stable-diffusion-turbo")
    enable_xformers: bool = bool(int(os.getenv("XFORMERS", "0")))
    warmup: bool = bool(int(os.getenv("WARMUP", "1")))
    server_port: int = int(os.getenv("PORT", "7860"))
    server_host: str = os.getenv("HOST", "0.0.0.0")
    share: bool = bool(int(os.getenv("SHARE", "1")))
