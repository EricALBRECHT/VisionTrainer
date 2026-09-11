"""Render tracked objects (IDs + optional trajectories) — separate from data."""

from __future__ import annotations

from PIL import Image, ImageDraw

from vision_trainer.inference.render import (
    AnnotationScale,
    BOX_COLORS,
    DISPLAY_MAX_WIDTH,
    DisplayTransform,
    _load_font,
    compute_annotation_style,
    prepare_display_image,
    scale_box_to_display,
)
from vision_trainer.tracking.models import TrackedObject


def format_track_label(obj: TrackedObject) -> str:
    conf_pct = int(round(float(obj.confidence) * 100.0))
    return f"{obj.class_name} · ID {obj.track_id} · {conf_pct}%"


def draw_tracked_objects(
    image: Image.Image,
    objects: list[TrackedObject],
    *,
    scale: AnnotationScale | str = "auto",
    fit_to_display: bool = False,
    show_trajectories: bool = False,
    max_display_width: int = DISPLAY_MAX_WIDTH,
) -> Image.Image:
    """Draw boxes + track labels (+ optional light trajectories)."""
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
    resolved = compute_annotation_style(width, height, scale)
    font = _load_font(resolved.font_size)
    pad_x = max(4, resolved.font_size // 5)
    pad_y = max(3, resolved.font_size // 6)
    gap = max(3, resolved.font_size // 8)

    for obj in objects:
        color = BOX_COLORS[int(obj.class_id) % len(BOX_COLORS)]
        x1, y1, x2, y2 = scale_box_to_display(*obj.bbox, transform)
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=resolved.line_width,
        )

        label = format_track_label(obj)
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

        if show_trajectories and len(obj.trajectory) >= 2:
            points = [
                (px * transform.scale_x, py * transform.scale_y)
                for px, py in obj.trajectory
            ]
            draw.line(points, fill=color, width=max(1, resolved.line_width - 1))

    return canvas


def draw_track_id_overlays(
    image: Image.Image,
    objects: list[TrackedObject],
    *,
    scale: AnnotationScale | str = "auto",
    fit_to_display: bool = False,
    show_trajectories: bool = False,
    max_display_width: int = DISPLAY_MAX_WIDTH,
) -> Image.Image:
    """
    Draw track ID badges + optional trajectories without re-drawing boxes.

    Useful when boxes/labels already come from the pipeline renderer.
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
    resolved = compute_annotation_style(width, height, scale)
    font = _load_font(max(12, resolved.font_size - 2))
    pad = max(2, resolved.font_size // 6)

    for obj in objects:
        color = BOX_COLORS[int(obj.class_id) % len(BOX_COLORS)]
        x1, y1, x2, y2 = scale_box_to_display(*obj.bbox, transform)
        label = f"ID {obj.track_id}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = max(1, text_bbox[2] - text_bbox[0])
        text_h = max(10, text_bbox[3] - text_bbox[1])
        tx = min(max(0, x1), max(0, width - text_w - 2 * pad))
        ty = min(y2 + pad, max(0, height - text_h - 2 * pad))
        draw.rectangle(
            [tx, ty, tx + text_w + 2 * pad, ty + text_h + 2 * pad],
            fill=color,
        )
        draw.text((tx + pad, ty + pad - text_bbox[1]), label, fill="white", font=font)

        if show_trajectories and len(obj.trajectory) >= 2:
            points = [
                (px * transform.scale_x, py * transform.scale_y)
                for px, py in obj.trajectory
            ]
            draw.line(points, fill=color, width=max(1, resolved.line_width - 1))

    return canvas
