from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from vision_trainer.yolo.models import SampleAnnotation
from vision_trainer.yolo.parser import label_path_for_image


class ImagePreviewError(Exception):
    """Raised when an image cannot be rendered in the preview."""


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


def parse_label_annotations(
    label_path: Path,
    class_names: dict[int, str],
) -> list[SampleAnnotation]:
    annotations: list[SampleAnnotation] = []
    if not label_path.is_file():
        return annotations

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            continue
        if class_id not in class_names:
            continue
        annotations.append(
            SampleAnnotation(
                class_id=class_id,
                class_name=class_names[class_id],
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )
    return annotations


def draw_annotations(image_path: Path, annotations: list[SampleAnnotation]) -> Image.Image:
    try:
        image = Image.open(image_path).convert("RGB")
    except (OSError, UnidentifiedImageError, SyntaxError) as exc:
        raise ImagePreviewError(f"Impossible d'afficher l'image {image_path.name} : {exc}") from exc

    draw = ImageDraw.Draw(image)
    width, height = image.size
    font = ImageFont.load_default()

    for annotation in annotations:
        color = BOX_COLORS[annotation.class_id % len(BOX_COLORS)]
        box_width = annotation.width * width
        box_height = annotation.height * height
        x1 = (annotation.x_center * width) - (box_width / 2)
        y1 = (annotation.y_center * height) - (box_height / 2)
        x2 = x1 + box_width
        y2 = y1 + box_height

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = annotation.class_name
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_height = text_bbox[3] - text_bbox[1]
        text_y = max(0, y1 - text_height - 2)
        draw.rectangle(
            [text_bbox[0], text_y, text_bbox[2], text_bbox[1] + (text_y - text_bbox[1])],
            fill=color,
        )
        draw.text((x1, text_y), label, fill="white", font=font)

    return image


def collect_sample_images(
    images_dir: Path,
    labels_dir: Path | None,
    class_names: dict[int, str],
    limit: int = 6,
) -> list[tuple[Path, list[SampleAnnotation]]]:
    samples: list[tuple[Path, list[SampleAnnotation]]] = []
    if not images_dir.is_dir() or labels_dir is None:
        return samples

    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        label_path = label_path_for_image(image_path, images_dir, labels_dir)
        annotations = parse_label_annotations(label_path, class_names)
        samples.append((image_path, annotations))
        if len(samples) >= limit:
            break
    return samples
