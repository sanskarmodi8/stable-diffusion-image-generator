"""UI for text to image generation section."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import gradio as gr


@dataclass
class Txt2ImgControls:
    """UI element references for the Text → Image tab.

    These allow the Presets tab to populate the fields programmatically.
    """

    prompt: gr.components.Textbox
    negative: gr.components.Textbox
    steps: gr.components.Slider
    guidance: gr.components.Slider
    width: gr.components.Slider
    height: gr.components.Slider
    seed: gr.components.Textbox


def build_txt2img_tab(handler: Callable[..., Tuple]) -> Txt2ImgControls:
    """Construct the Text → Image tab and bind the Generate button.

    Args:
        handler: Function that performs txt2img and returns (image, metadata).

    Returns:
        A Txt2ImgControls instance containing references to all UI controls.
    """
    with gr.Tab("Text → Image"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="A futuristic city at dusk, cinematic lighting",
                )
                negative = gr.Textbox(
                    label="Negative prompt",
                    placeholder="low quality, blurry, extra limbs",
                )

                steps = gr.Slider(
                    minimum=10,
                    maximum=50,
                    value=30,
                    step=1,
                    label="Steps",
                )
                gr.Markdown(
                    "More steps → finer detail, slower runtime. 20–40 is typical.",
                )

                guidance = gr.Slider(
                    minimum=1,
                    maximum=15,
                    value=7.5,
                    step=0.5,
                    label="Guidance Scale (CFG)",
                )
                gr.Markdown(
                    "Higher values make generation match the prompt more strictly. "
                    "7–9 is a common range.",
                )

                width = gr.Slider(
                    minimum=256,
                    maximum=768,
                    value=512,
                    step=64,
                    label="Width",
                )
                height = gr.Slider(
                    minimum=256,
                    maximum=768,
                    value=512,
                    step=64,
                    label="Height",
                )

                seed = gr.Textbox(
                    label="Seed (optional)",
                    value="",
                    placeholder="Leave empty for random",
                )

                generate_button = gr.Button("Generate")

            with gr.Column():
                out_image = gr.Image(label="Output")
                out_meta = gr.JSON(label="Metadata (JSON)")

        generate_button.click(
            fn=handler,
            inputs=[prompt, negative, steps, guidance, width, height, seed],
            outputs=[out_image, out_meta],
        )

    return Txt2ImgControls(
        prompt=prompt,
        negative=negative,
        steps=steps,
        guidance=guidance,
        width=width,
        height=height,
        seed=seed,
    )
