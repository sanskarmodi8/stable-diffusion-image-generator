"""UI for History section."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image

from src.sdgen.utils.common import short_prompt
from src.sdgen.utils.history import (
    delete_history_entry,
    list_history,
    load_entry,
)
from src.sdgen.utils.logger import get_logger

logger = get_logger(__name__)


# Internal helpers


def _label(entry: Dict[str, Any]) -> str:
    """Human-readable dropdown label."""
    ts = entry.get("timestamp", "")[:19].replace("T", " ")
    mode = entry.get("mode", "unknown")
    prompt = short_prompt(entry.get("prompt", ""), 60)
    return f"{ts} — {mode} — {prompt}" if prompt else f"{ts} — {mode}"


def _build_index(limit: int = 500) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Load history index → (ids, labels, raw entries)."""
    entries = list_history(limit)
    ids = [e.get("id", "") for e in entries]
    labels = [_label(e) for e in entries]
    return ids, labels, entries


def _id_from_label(label: str, entries: List[Dict[str, Any]]) -> Optional[str]:
    """Resolve entry ID from label text."""
    for e in entries:
        if _label(e) == label:
            return e.get("id")
    return None


# Operations


def load_from_dropdown(selected_label: str, entries: List[Dict[str, Any]]):
    """Load a history entry from dropdown."""
    if not selected_label:
        raise gr.Error("No entry selected.")

    entry_id = _id_from_label(selected_label, entries)
    if not entry_id:
        raise gr.Error("Entry not found.")

    data = load_entry(entry_id)
    if not data:
        raise gr.Error("Entry JSON missing.")

    thumb_path = data.get("thumbnail")
    img = Image.open(thumb_path) if thumb_path else None

    return img, data


def refresh_history():
    """Refresh dropdown + state.

    Clear output.
    """
    _, labels, entries = _build_index()
    if labels:
        dd = gr.update(choices=labels, value=labels[0])
    else:
        dd = gr.update(choices=[], value=None)

    return dd, entries, None, {}


def delete_entry(selected_label: str, entries: List[Dict[str, Any]]):
    """Delete and refresh UI."""
    if not selected_label:
        raise gr.Error("Select an entry first.")

    entry_id = _id_from_label(selected_label, entries)
    if not entry_id:
        raise gr.Error("Entry not found.")

    ok = delete_history_entry(entry_id)
    if not ok:
        raise gr.Error("Delete failed.")

    _, labels, new_entries = _build_index()

    if labels:
        dd = gr.update(choices=labels, value=labels[0])
    else:
        dd = gr.update(choices=[], value=None)

    return None, {}, dd, new_entries


# UI


def build_history_tab() -> None:
    """History tab: dropdown, load button, delete, refresh."""
    _, labels, entries = _build_index()
    initial = labels[0] if labels else None

    with gr.Tab("History"):
        with gr.Row():
            # Left panel: controls
            with gr.Column(scale=1):
                dropdown = gr.Dropdown(
                    label="History entries",
                    choices=labels,
                    value=initial,
                    interactive=True,
                )

                load_btn = gr.Button("Load entry")
                refresh_btn = gr.Button("Refresh")
                delete_btn = gr.Button("Delete selected", variant="stop")

            # Right panel: output
            with gr.Column(scale=2):
                thumb = gr.Image(
                    label="Thumbnail",
                    show_label=True,
                    type="pil",
                )
                meta = gr.JSON(
                    label="Metadata",
                )

        state = gr.State(entries)

        # Events

        load_btn.click(
            fn=load_from_dropdown,
            inputs=[dropdown, state],
            outputs=[thumb, meta],
        )

        refresh_btn.click(
            fn=refresh_history,
            inputs=None,
            outputs=[dropdown, state, thumb, meta],
        )

        delete_btn.click(
            fn=delete_entry,
            inputs=[dropdown, state],
            outputs=[thumb, meta, dropdown, state],
        )
