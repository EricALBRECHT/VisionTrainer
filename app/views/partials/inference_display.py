"""Shared Streamlit control for inference image display size (visual only)."""

from __future__ import annotations

import streamlit as st
from PIL import Image

from vision_trainer.inference.render import (
    DEFAULT_INFERENCE_DISPLAY_FRACTION,
    INFERENCE_DISPLAY_SIZE_CHOICES,
    inference_display_max_width,
    preview_image_for_ui,
)


def render_inference_display_size_selector(*, key: str) -> float:
    """Return selected display fraction (0.25 … 1.0). Default 50 %."""
    labels = [label for label, _ in INFERENCE_DISPLAY_SIZE_CHOICES]
    fractions = {label: value for label, value in INFERENCE_DISPLAY_SIZE_CHOICES}
    default_label = next(
        label
        for label, value in INFERENCE_DISPLAY_SIZE_CHOICES
        if value == DEFAULT_INFERENCE_DISPLAY_FRACTION
    )
    default_index = labels.index(default_label)
    selected = st.selectbox(
        "Taille d'affichage",
        options=labels,
        index=default_index,
        key=key,
        help=(
            "Redimensionne uniquement l'aperçu Streamlit. "
            "L'inférence utilise toujours l'image originale."
        ),
    )
    return float(fractions.get(selected, DEFAULT_INFERENCE_DISPLAY_FRACTION))


def show_inference_preview(
    image: Image.Image | bytes,
    *,
    display_fraction: float,
    caption: str | None = None,
) -> None:
    """Show an image (or JPEG bytes) at the selected visual scale."""
    if isinstance(image, (bytes, bytearray)):
        from io import BytesIO

        pil = Image.open(BytesIO(image)).convert("RGB")
    else:
        pil = image
    preview = preview_image_for_ui(pil, display_fraction=display_fraction)
    st.image(preview, caption=caption, use_container_width=False)


def inference_annotation_max_width(display_fraction: float) -> int:
    """Same max width used for annotated draw_* previews."""
    return inference_display_max_width(display_fraction)
