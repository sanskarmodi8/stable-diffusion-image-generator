"""UI for upscaler section."""

from __future__ import annotations

from typing import Callable

import gradio as gr


def build_upscaler_tab(handler: Callable[..., tuple]) -> None:
    """Build the Upscaler tab and wire it to the given handler."""
    with gr.Tab("Upscaler"):
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Image to Upscale",
                    type="numpy",
                )
                scale = gr.Radio(
                    choices=["2.0", "4.0"],
                    value="2.0",
                    label="Upscale Factor",
                )
                upscale_button = gr.Button("Upscale")

            with gr.Column():
                out_image = gr.Image(label="Upscaled Image")
                out_meta = gr.JSON(
                    label="Metadata (JSON)",
                )

        upscale_button.click(
            fn=handler,
            inputs=[input_image, scale],
            outputs=[out_image, out_meta],
        )
