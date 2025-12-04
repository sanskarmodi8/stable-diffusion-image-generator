from __future__ import annotations

from .history_tab import build_history_tab
from .img2img_tab import build_img2img_tab
from .presets_tab import build_presets_tab
from .txt2img_tab import build_txt2img_tab
from .upscaler_tab import build_upscaler_tab

__all__ = [
    "build_txt2img_tab",
    "build_img2img_tab",
    "build_upscaler_tab",
    "build_presets_tab",
    "build_history_tab",
]
