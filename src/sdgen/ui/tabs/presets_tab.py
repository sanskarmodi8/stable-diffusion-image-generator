"""UI for presets section."""

from __future__ import annotations

from typing import Any, Tuple

import gradio as gr

from src.sdgen.presets.styles import get_preset, list_presets
from src.sdgen.ui.tabs.img2img_tab import Img2ImgControls
from src.sdgen.ui.tabs.txt2img_tab import Txt2ImgControls


def apply_preset(preset_name: Any) -> Tuple[Any, ...]:
    """Return values to populate txt2img and img2img controls.

    Args:
        preset_name: A string or a one-element list representing the preset key.

    Returns:
        A tuple with values mapped to Text→Image and Image→Image UI controls.
    """
    # unwrap dropdown list behavior
    if isinstance(preset_name, (list, tuple)):
        preset_name = preset_name[0] if preset_name else None

    if not preset_name:
        raise gr.Error("Select a preset first.")

    preset = get_preset(str(preset_name))
    if preset is None:
        raise gr.Error("Invalid preset selected.")

    prompt = preset.get("prompt", "")
    negative = preset.get("negative_prompt", "")

    # only return data; UI wiring chooses what to set
    status_msg = f"Applied preset: {preset_name}"

    return (
        # txt2img
        prompt,
        negative,
        # img2img
        prompt,
        negative,
        # status
        status_msg,
    )


def build_presets_tab(
    txt_controls: Txt2ImgControls,
    img_controls: Img2ImgControls,
) -> None:
    """Construct the Presets tab and link values to both txt2img and img2img controls.

    Args:
        txt_controls: References to Text→Image input controls.
        img_controls: References to Image→Image input controls.
    """
    with gr.Tab("Presets"):
        with gr.Row():
            with gr.Column():
                preset_name = gr.Dropdown(
                    choices=list_presets(),
                    label="Select style",
                )
                apply_button = gr.Button("Apply Preset")
                status_box = gr.Markdown("")

            with gr.Column():
                gr.Markdown(
                    "Applying a preset fills prompt, negative prompt, steps, "
                    "guidance, and resolution for both **Text → Image** "
                    "and **Image → Image** tabs.",
                )

        apply_button.click(
            fn=apply_preset,
            inputs=[preset_name],
            outputs=[
                # txt2img
                txt_controls.prompt,
                txt_controls.negative,
                # img2img
                img_controls.prompt,
                img_controls.negative,
                # status markdown
                status_box,
            ],
        )
