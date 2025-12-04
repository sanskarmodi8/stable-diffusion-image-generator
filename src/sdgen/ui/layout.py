"""UI layout builder for the Stable Diffusion Gradio app."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import gradio as gr

from sdgen.sd.generator import generate_image
from sdgen.sd.img2img import generate_img2img
from sdgen.sd.models import Img2ImgConfig, Txt2ImgConfig
from sdgen.ui.tabs import (
    build_history_tab,
    build_img2img_tab,
    build_presets_tab,
    build_txt2img_tab,
    build_upscaler_tab,
)
from sdgen.ui.tabs.img2img_tab import Img2ImgControls
from sdgen.ui.tabs.txt2img_tab import Txt2ImgControls
from sdgen.upscaler.upscaler import Upscaler
from sdgen.utils.common import pretty_json, to_pil
from sdgen.utils.history import save_history_entry
from sdgen.utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_seed(value: Any) -> int | None:
    """Return integer seed if valid, otherwise None."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        logger.warning("Invalid seed input: %s", value)
        return None


def _txt2img_handler(
    pipe: Any,
    prompt: str,
    negative: str,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    seed: Any,
) -> Tuple[Any, str]:
    """Run text-to-image generation."""
    cfg = Txt2ImgConfig(
        prompt=prompt or "",
        negative_prompt=negative or "",
        steps=int(steps),
        guidance_scale=float(guidance),
        width=int(width),
        height=int(height),
        seed=_resolve_seed(seed),
        device=pipe.device.type,
    )

    image, meta = generate_image(pipe, cfg)

    try:
        save_history_entry(meta, image)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to save history entry: %s", exc)

    return image, pretty_json(meta.to_dict())


def _img2img_handler(
    pipe: Any,
    input_image: Any,
    prompt: str,
    negative: str,
    strength: float,
    steps: int,
    guidance: float,
    seed: Any,
) -> Tuple[Any, str]:
    """Run image-to-image generation."""
    if input_image is None:
        raise gr.Error("Upload an image to continue.")

    pil_image = to_pil(input_image)

    cfg = Img2ImgConfig(
        prompt=prompt or "",
        negative_prompt=negative or "",
        strength=float(strength),
        steps=int(steps),
        guidance_scale=float(guidance),
        width=pil_image.width,
        height=pil_image.height,
        seed=_resolve_seed(seed),
        device=pipe.device.type,
    )

    image, meta = generate_img2img(pipe, cfg, pil_image)

    try:
        save_history_entry(meta, image)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to save history entry: %s", exc)

    return image, pretty_json(meta.to_dict())


def _upscale_handler(
    input_image: Any,
    scale: str,
) -> Tuple[Any, str]:
    """Run image upscaling."""
    if input_image is None:
        raise gr.Error("Upload an image to continue.")

    pil_image = to_pil(input_image)

    # scale is str → convert to int
    try:
        scale_int = int(float(scale))
    except Exception as exc:  # noqa: BLE001
        raise gr.Error("Scale must be numeric (2 or 4).") from exc

    upscaler = Upscaler(scale=scale_int, prefer="ncnn")
    out_image = upscaler.upscale(pil_image)

    meta: Dict[str, Any] = {
        "mode": "upscale",
        "scale": scale_int,
        "width": out_image.width,
        "height": out_image.height,
    }

    try:
        save_history_entry(meta, out_image)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to save history entry: %s", exc)

    return out_image, pretty_json(meta)


def build_ui(txt2img_pipe: Any, img2img_pipe: Any) -> gr.Blocks:
    """Build the entire Gradio UI."""
    with gr.Blocks() as demo:
        gr.Markdown(
            "# Stable Diffusion Generator\n"
            "Clean, local Stable \
            Diffusion toolkit."
        )

        txt_controls: Txt2ImgControls = build_txt2img_tab(
            handler=lambda *args: _txt2img_handler(txt2img_pipe, *args),
        )

        img_controls: Img2ImgControls = build_img2img_tab(
            handler=lambda *args: _img2img_handler(img2img_pipe, *args),
        )

        build_upscaler_tab(
            handler=_upscale_handler,
        )

        build_presets_tab(
            txt_controls=txt_controls,
            img_controls=img_controls,
        )

        build_history_tab()

        gr.Markdown(
            "### Notes\n"
            "- Seeds left blank will be randomized.\n"
            "- Use **History → Refresh History** if new thumbnails do not appear.\n"
            "- Presets apply to both **Text → Image** and **Image → Image** tabs.\n"
        )

    return demo
