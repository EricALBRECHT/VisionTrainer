"""Dataset preview: draw segmentation polygons on sample images."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from vision_trainer.inference.render import BOX_COLORS
from vision_trainer.segment.labels import (
    parse_segment_label_file,
    polygon_bbox_xyxy,
    polygon_to_pixel_points,
)
from vision_trainer.yolo.parser import label_path_for_image


class SegmentPreviewError(Exception):
    """Raised when a segmentation sample cannot be previewed."""


def draw_segment_preview(
    image_path: Path,
    labels_dir: Path,
    images_dir: Path,
    class_names: dict[int, str],
    *,
    show_bbox: bool = True,
) -> Image.Image:
    """
    Draw polygons (and optional AABB) on a copy of the image.

    Never modifies the original file.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except (OSError, UnidentifiedImageError, SyntaxError) as exc:
        raise SegmentPreviewError(
            f"Impossible d'afficher {image_path.name} : {exc}"
        ) from exc

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = canvas.size
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None

    label_path = label_path_for_image(image_path, images_dir, labels_dir)
    instances, _errors = parse_segment_label_file(label_path, class_names=class_names)

    for instance in instances:
        color = BOX_COLORS[instance.class_id % len(BOX_COLORS)]
        # Convert hex to RGBA with alpha for fill
        fill = _hex_to_rgba(color, alpha=80)
        outline = _hex_to_rgba(color, alpha=255)
        pixels = polygon_to_pixel_points(
            instance.points, image_width=width, image_height=height
        )
        if len(pixels) < 3:
            continue
        draw.polygon(pixels, outline=outline, fill=fill)
        if show_bbox:
            bbox = polygon_bbox_xyxy(pixels)
            if bbox is not None:
                draw.rectangle(bbox, outline=outline, width=2)
        name = class_names.get(instance.class_id, str(instance.class_id))
        x0, y0 = pixels[0]
        if font is not None:
            draw.text((x0 + 2, y0 + 2), name, fill=outline, font=font)
        else:
            draw.text((x0 + 2, y0 + 2), name, fill=outline)

    return canvas.convert("RGB")


def _hex_to_rgba(color: str, *, alpha: int) -> tuple[int, int, int, int]:
    value = color.lstrip("#")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return r, g, b, max(0, min(255, alpha))
