from __future__ import annotations

import io
from pathlib import Path

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


def format_detection_label(detection: Detection) -> str:
    percent = int(round(detection.confidence * 100))
    return f"{detection.class_name} {percent}%"


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    """Draw pixel-space bounding boxes with class + confidence labels."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for detection in detections:
        color = BOX_COLORS[detection.class_id % len(BOX_COLORS)]
        x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = format_detection_label(detection)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_height = text_bbox[3] - text_bbox[1]
        text_width = text_bbox[2] - text_bbox[0]
        text_y = max(0, y1 - text_height - 4)
        draw.rectangle(
            [x1, text_y, x1 + text_width + 4, text_y + text_height + 2],
            fill=color,
        )
        draw.text((x1 + 2, text_y), label, fill="white", font=font)

    return annotated


def annotated_image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def build_download_filename(original_name: str) -> str:
    stem = Path(original_name).stem or "image"
    return f"prediction_{stem}.jpg"
