"""Preset configurations for text-to-image generation.

This module defines a collection of named presets including prompt,
negative prompt, sampler parameters, and recommended resolutions.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Global preset registry: {preset_name: parameters}
PRESETS: Dict[str, Dict[str, Any]] = {
    "Realistic Photo": {
        "prompt": (
            "ultra realistic, 35mm photography, \
photorealistic, cinematic lighting"
        ),
        "negative_prompt": "low quality, blurry, deformed, extra limbs",
        "tags": ["realistic", "photo"],
    },
    "Anime": {
        "prompt": (
            "high quality anime, clean lines, vibrant colors, \
soft rim lighting, studio lighting"
        ),
        "negative_prompt": "blurry, low detail, mutation, deformed",
        "tags": ["anime", "stylized"],
    },
    "Cinematic / Moody": {
        "prompt": (
            "dramatic cinematic lighting, moody, film grain, \
Kodak Portra, filmic color grading"
        ),
        "negative_prompt": "oversaturated, low detail, flat lighting",
        "tags": ["cinematic", "moody"],
    },
    "Oil Painting / Classic Art": {
        "prompt": (
            "oil painting, impasto brush strokes, classical \
lighting, Rembrandt style"
        ),
        "negative_prompt": "blurry, cartoonish, digital artifacts",
        "tags": ["art", "oil", "painterly"],
    },
    "Cyberpunk / Neon": {
        "prompt": (
            "cyberpunk city, neon reflections, wet streets, \
high detail, synthwave aesthetic"
        ),
        "negative_prompt": "low detail, daytime, blurry",
        "tags": ["cyberpunk", "neon"],
    },
}


def get_preset(name: str) -> Dict[str, Any] | None:
    """Return a shallow copy of a preset by name."""
    data = PRESETS.get(name)
    return dict(data) if data else None


def list_presets() -> List[str]:
    """List preset names in a stable UI order."""
    # Avoid unexpected reordering: use insertion order
    return list(PRESETS.keys())
