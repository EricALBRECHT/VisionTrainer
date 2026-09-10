"""Readable segmentation overlay rendering (display-canvas based)."""

from __future__ import annotations

from PIL import Image, ImageDraw

from vision_trainer.inference.render import (
    BOX_COLORS,
    DISPLAY_MAX_WIDTH,
    AnnotationScale,
    AnnotationStyle,
    DisplayTransform,
    compute_annotation_style,
    prepare_display_image,
    scale_box_to_display,
    _load_font,
)
from vision_trainer.segment.models import SegmentationResult


def scale_polygon_to_display(
    polygon: tuple[tuple[float, float], ...],
    transform: DisplayTransform,
) -> list[tuple[float, float]]:
    return [(x * transform.scale_x, y * transform.scale_y) for x, y in polygon]


def draw_segmentation_result(
    image: Image.Image,
    result: SegmentationResult,
    *,
    show_masks: bool = True,
    show_contours: bool = True,
    show_labels: bool = True,
    show_boxes: bool = False,
    mask_opacity: float = 0.40,
    scale: AnnotationScale | str = "auto",
    style: AnnotationStyle | None = None,
    max_display_width: int = DISPLAY_MAX_WIDTH,
    fit_to_display: bool = True,
) -> Image.Image:
    """
    Draw masks / contours / labels on a display-sized canvas.

    Coordinates in ``result`` are assumed to be in original image space.
    Missing polygons still allow bbox / label rendering.
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

    overlay = canvas.convert("RGBA")
    draw_overlay = ImageDraw.Draw(overlay, "RGBA")
    width, height = canvas.size
    resolved = style if style is not None else compute_annotation_style(width, height, scale)
    font = _load_font(resolved.font_size)
    pad_x = max(4, resolved.font_size // 5)
    pad_y = max(3, resolved.font_size // 6)
    gap = max(3, resolved.font_size // 8)
    alpha = max(0, min(255, int(round(float(mask_opacity) * 255))))

    label_jobs: list[tuple[float, float, int, int, str, str, tuple[int, int, int, int]]] = []

    for item in result.instances:
        color = BOX_COLORS[item.class_id % len(BOX_COLORS)]
        rgb = _hex_to_rgb(color)
        fill = (*rgb, alpha)
        outline = (*rgb, 255)

        polygon = scale_polygon_to_display(item.polygon, transform)
        has_poly = len(polygon) >= 3

        if show_masks and has_poly:
            draw_overlay.polygon(polygon, fill=fill)
        if show_contours and has_poly:
            draw_overlay.line(
                polygon + [polygon[0]],
                fill=outline,
                width=max(2, resolved.line_width),
            )

        need_box = show_boxes or (not has_poly and (show_masks or show_contours))
        if need_box:
            x1, y1, x2, y2 = scale_box_to_display(
                item.x1, item.y1, item.x2, item.y2, transform
            )
            draw_overlay.rectangle(
                [x1, y1, x2, y2],
                outline=outline,
                width=resolved.line_width,
            )

        if show_labels:
            label = f"{item.class_name} {item.confidence:.2f}"
            bx1, by1, _, _ = scale_box_to_display(
                item.x1, item.y1, item.x2, item.y2, transform
            )
            # Probe text size on RGB canvas
            probe = ImageDraw.Draw(canvas)
            text_bbox = probe.textbbox((0, 0), label, font=font)
            text_width = max(1, text_bbox[2] - text_bbox[0])
            text_height = max(resolved.font_size, text_bbox[3] - text_bbox[1])
            box_w = text_width + 2 * pad_x
            box_h = text_height + 2 * pad_y
            text_y = by1 - box_h - gap
            if text_y < 0:
                text_y = min(by1 + gap, max(0, height - box_h))
            text_x = min(max(0, bx1), max(0, width - box_w))
            label_jobs.append((text_x, text_y, box_w, box_h, label, color, text_bbox))

    composed = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw_final = ImageDraw.Draw(composed)
    for text_x, text_y, box_w, box_h, label, color, text_bbox in label_jobs:
        draw_final.rectangle(
            [text_x, text_y, text_x + box_w, text_y + box_h],
            fill=color,
        )
        draw_final.text(
            (text_x + pad_x, text_y + pad_y - text_bbox[1]),
            label,
            fill="white",
            font=font,
        )
    return composed


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
