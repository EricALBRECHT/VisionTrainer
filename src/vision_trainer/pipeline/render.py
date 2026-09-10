"""Readable pipeline overlay: boxes, classification labels, segmentation masks."""

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
from vision_trainer.pipeline.models import PipelineResult
from vision_trainer.segment.render import scale_polygon_to_display


def draw_pipeline_result(
    image: Image.Image,
    result: PipelineResult,
    *,
    show_boxes: bool = True,
    show_classification: bool = True,
    show_masks: bool = True,
    show_contours: bool = True,
    show_seg_labels: bool = False,
    mask_opacity: float = 0.40,
    scale: AnnotationScale | str = "auto",
    style: AnnotationStyle | None = None,
    max_display_width: int = DISPLAY_MAX_WIDTH,
    fit_to_display: bool = True,
) -> Image.Image:
    """
    Draw enriched pipeline results on a display-sized canvas.

    Detection coordinates and segmentation polygons are in original image space.
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

    for item in result.items:
        detection = item.detection
        color = BOX_COLORS[detection.class_id % len(BOX_COLORS)]
        rgb = _hex_to_rgb(color)
        outline = (*rgb, 255)

        if show_masks or show_contours:
            for mask in item.segmentations:
                seg_color = BOX_COLORS[mask.class_id % len(BOX_COLORS)]
                seg_rgb = _hex_to_rgb(seg_color)
                poly = scale_polygon_to_display(mask.polygon_global, transform)
                if show_masks and len(poly) >= 3:
                    draw_overlay.polygon(poly, fill=(*seg_rgb, alpha))
                if show_contours and len(poly) >= 3:
                    draw_overlay.line(
                        poly + [poly[0]],
                        fill=(*seg_rgb, 255),
                        width=max(2, resolved.line_width),
                    )
                if show_seg_labels and mask.bbox_global is not None:
                    sx1, sy1, _, _ = scale_box_to_display(
                        *mask.bbox_global, transform
                    )
                    seg_label = f"{mask.class_name} {mask.confidence:.2f}"
                    label_jobs.append(
                        _make_label_job(
                            canvas,
                            font,
                            seg_label,
                            seg_color,
                            sx1,
                            sy1,
                            width,
                            height,
                            pad_x,
                            pad_y,
                            gap,
                            resolved.font_size,
                        )
                    )

        if show_boxes:
            x1, y1, x2, y2 = scale_box_to_display(
                detection.x1,
                detection.y1,
                detection.x2,
                detection.y2,
                transform,
            )
            draw_overlay.rectangle(
                [x1, y1, x2, y2],
                outline=outline,
                width=resolved.line_width,
            )

        if show_classification or not item.refined:
            label = item.display_label if show_classification else (
                f"{detection.class_name} {detection.confidence:.2f}"
            )
            bx1, by1, _, _ = scale_box_to_display(
                detection.x1,
                detection.y1,
                detection.x2,
                detection.y2,
                transform,
            )
            label_jobs.append(
                _make_label_job(
                    canvas,
                    font,
                    label,
                    color,
                    bx1,
                    by1,
                    width,
                    height,
                    pad_x,
                    pad_y,
                    gap,
                    resolved.font_size,
                )
            )

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


def _make_label_job(
    canvas: Image.Image,
    font,
    label: str,
    color: str,
    bx1: float,
    by1: float,
    width: int,
    height: int,
    pad_x: int,
    pad_y: int,
    gap: int,
    font_size: int,
) -> tuple[float, float, int, int, str, str, tuple[int, int, int, int]]:
    probe = ImageDraw.Draw(canvas)
    text_bbox = probe.textbbox((0, 0), label, font=font)
    text_width = max(1, text_bbox[2] - text_bbox[0])
    text_height = max(font_size, text_bbox[3] - text_bbox[1])
    box_w = text_width + 2 * pad_x
    box_h = text_height + 2 * pad_y
    text_y = by1 - box_h - gap
    if text_y < 0:
        text_y = min(by1 + gap, max(0, height - box_h))
    text_x = min(max(0, bx1), max(0, width - box_w))
    return text_x, text_y, box_w, box_h, label, color, text_bbox


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
