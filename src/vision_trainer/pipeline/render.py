"""Rendering helpers for enriched pipeline detections."""

from __future__ import annotations

from PIL import Image, ImageDraw

from vision_trainer.inference.render import (
    BOX_COLORS,
    DISPLAY_MAX_WIDTH,
    AnnotationScale,
    AnnotationStyle,
    compute_annotation_style,
    prepare_display_image,
    scale_box_to_display,
    _load_font,
)
from vision_trainer.inference.render import DisplayTransform
from vision_trainer.pipeline.models import EnrichedDetection, PipelineResult


def draw_pipeline_result(
    image: Image.Image,
    result: PipelineResult,
    *,
    scale: AnnotationScale | str = "auto",
    style: AnnotationStyle | None = None,
    max_display_width: int = DISPLAY_MAX_WIDTH,
    fit_to_display: bool = True,
) -> Image.Image:
    """
    Draw enriched labels using the V1 readable annotation path.

    Uses ``EnrichedDetection.display_label`` directly (does not append
    detection confidence a second time via ``format_detection_label``).
    """
    if fit_to_display:
        canvas, transform = prepare_display_image(image, max_width=max_display_width)
    else:
        canvas = image.convert("RGB").copy()
        transform = DisplayTransform(
            original_width=canvas.width,
            original_height=canvas.height,
            display_width=canvas.width,
            display_height=canvas.height,
            scale_x=1.0,
            scale_y=1.0,
        )

    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    resolved = style if style is not None else compute_annotation_style(width, height, scale)
    font = _load_font(resolved.font_size)
    pad_x = max(4, resolved.font_size // 5)
    pad_y = max(3, resolved.font_size // 6)
    gap = max(3, resolved.font_size // 8)

    for item in result.items:
        detection = item.detection
        color = BOX_COLORS[detection.class_id % len(BOX_COLORS)]
        x1, y1, x2, y2 = scale_box_to_display(
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
            transform,
        )
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=resolved.line_width,
        )

        label = item.display_label
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = max(1, text_bbox[2] - text_bbox[0])
        text_height = max(resolved.font_size, text_bbox[3] - text_bbox[1])
        box_w = text_width + 2 * pad_x
        box_h = text_height + 2 * pad_y

        text_y = y1 - box_h - gap
        if text_y < 0:
            text_y = min(y1 + gap, max(0, height - box_h))
        text_x = min(max(0, x1), max(0, width - box_w))

        draw.rectangle(
            [text_x, text_y, text_x + box_w, text_y + box_h],
            fill=color,
        )
        draw.text(
            (text_x + pad_x, text_y + pad_y - text_bbox[1]),
            label,
            fill="white",
            font=font,
        )

    return canvas
