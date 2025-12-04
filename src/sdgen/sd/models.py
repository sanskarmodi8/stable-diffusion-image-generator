"""Configuration dataclasses for Stable Diffusion execution and history storage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class Txt2ImgConfig:
    """Configuration for text-to-image generation.

    Attributes:
        prompt: Positive prompt text.
        negative_prompt: Negative prompt text.
        steps: Number of diffusion steps.
        guidance_scale: Classifier-free guidance scale.
        width: Requested image width.
        height: Requested image height.
        seed: Optional random seed.
        device: Target torch device ("cuda" or "cpu").
    """

    prompt: str
    negative_prompt: str = ""
    steps: int = 30
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    seed: Optional[int] = None
    device: str = "cuda"


@dataclass
class Img2ImgConfig:
    """Configuration for image-to-image generation.

    Attributes:
        prompt: Positive prompt text.
        init_image_path: Optional file path to source image.
        negative_prompt: Negative prompt text.
        strength: Img2Img blend strength in (0, 1].
        steps: Number of diffusion steps.
        guidance_scale: CFG scale.
        width: Requested image width.
        height: Requested image height.
        seed: Optional random seed.
        device: Target device.
    """

    prompt: str
    init_image_path: Optional[str] = None
    negative_prompt: str = ""
    strength: float = 0.7
    steps: int = 30
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    seed: Optional[int] = None
    device: str = "cuda"


@dataclass
class GenerationMetadata:
    """Output metadata for a generated image.

    Attributes:
        mode: Generation mode ("txt2img", "img2img", "upscale", ...).
        prompt: Prompt text.
        negative_prompt: Negative prompt text.
        steps: Number of diffusion steps.
        guidance_scale: CFG scale.
        width: Output width.
        height: Output height.
        seed: Resolved random seed.
        strength: Img2Img strength; None for Txt2Img.
        elapsed_seconds: Wall-clock runtime.
        timestamp: UTC timestamp.
        id: Unique entry ID.
        thumbnail: Local thumbnail path.
        full_image: Local full-size image path.
    """

    mode: str
    prompt: str
    negative_prompt: str = ""
    steps: int = 30
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    seed: Optional[int] = None
    strength: Optional[float] = None
    elapsed_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    id: Optional[str] = None
    thumbnail: Optional[str] = None
    full_image: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a dict representation excluding None values."""
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}


@dataclass
class HistorySummary:
    """Minimal entry used for UI history lists."""

    id: str
    prompt: str
    mode: str
    seed: Optional[int]
    width: int
    height: int
    timestamp: str
    thumbnail: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dict representation."""
        return asdict(self)
