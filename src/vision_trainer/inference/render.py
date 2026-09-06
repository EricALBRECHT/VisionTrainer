from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

from vision_trainer.inference.models import Detection

BOX_COLORS = [
    "#E6194B",
    "#3CB44B",
    "#4363D8",
    "#F58231",
    "#911EB4",
    "#42D4F4",
    "#F032E6",
    "#BFEF45",
]

AnnotationScale = Literal["auto", "small", "medium", "large"]

# Stable UI order: French label → scale key.
ANNOTATION_SCALE_OPTIONS: tuple[tuple[str, AnnotationScale], ...] = (
    ("Auto", "auto"),
    ("Petite", "small"),
    ("Moyenne", "medium"),
    ("Grande", "large"),
)

# Multipliers applied on top of the Auto formula (Ultralytics-style).
_SCALE_FACTORS: dict[str, float] = {
    "auto": 1.0,
    "small": 0.7,
    "medium": 1.25,
    "large": 1.75,
}

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
)


@dataclass(frozen=True)
class AnnotationStyle:
    """Resolved drawing parameters for one image/frame size."""

    line_width: int
    font_size: int


def compute_annotation_style(
    width: int,
    height: int,
    scale: AnnotationScale | str = "auto",
) -> AnnotationStyle:
    """
    Derive box thickness and label font size from image resolution.

    Formula mirrors Ultralytics ``Annotator`` defaults, scaled by the UI preset:
    - ``line_width = max(round(mean_side * 0.003 * factor), 2)``
    - ``font_size  = max(round(mean_side * 0.035 * factor), 12)``
    """
    w = max(1, int(width))
    h = max(1, int(height))
    mean_side = (w + h) / 2.0
    factor = _SCALE_FACTORS.get(str(scale).strip().lower(), 1.0)

    def _round_half_up(value: float) -> int:
        return int(value + 0.5)

    line_width = max(2, _round_half_up(mean_side * 0.003 * factor))
    font_size = max(12, _round_half_up(mean_side * 0.035 * factor))
    # Keep boxes readable but not cartoonishly thick vs. the text.
    line_width = min(line_width, max(2, font_size // 3))
    return AnnotationStyle(line_width=line_width, font_size=font_size)


@lru_cache(maxsize=64)
def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load a scalable TTF when available; fall back to Pillow default."""
    size = max(8, int(size))
    for candidate in _FONT_CANDIDATES:
        path = Path(candidate)
        if not path.is_file():
            continue
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            continue
    try:
        # Pillow ≥10 accepts size= for the bitmap default font.
        return ImageFont.load_default(size=size)  # type: ignore[call-arg]
    except TypeError:
        return ImageFont.load_default()


def format_detection_label(detection: Detection) -> str:
    percent = int(round(detection.confidence * 100))
    return f"{detection.class_name} {percent}%"


def draw_detections(
    image: Image.Image,
    detections: list[Detection],
    *,
    scale: AnnotationScale | str = "auto",
    style: AnnotationStyle | None = None,
) -> Image.Image:
    """Draw pixel-space bounding boxes with class + confidence labels."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size
    resolved = style if style is not None else compute_annotation_style(width, height, scale)
    font = _load_font(resolved.font_size)
    pad_x = max(2, resolved.font_size // 6)
    pad_y = max(1, resolved.font_size // 8)
    gap = max(2, resolved.font_size // 10)

    for detection in detections:
        color = BOX_COLORS[detection.class_id % len(BOX_COLORS)]
        x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=resolved.line_width,
        )

        label = format_detection_label(detection)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
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

    return annotated


def annotated_image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def build_download_filename(original_name: str) -> str:
    stem = Path(original_name).stem or "image"
    return f"prediction_{stem}.jpg"
