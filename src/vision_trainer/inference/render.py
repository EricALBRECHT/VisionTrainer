from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
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

# Multipliers on display-based Auto sizes.
_SCALE_FACTORS: dict[str, float] = {
    "auto": 1.0,
    "small": 0.75,
    "medium": 1.2,
    "large": 1.5,
}

# Max width of the image used for on-screen / downloadable annotations.
# Inference still runs on the original; only the visualization is resized.
DISPLAY_MAX_WIDTH = 1280

# Target Auto sizes when the display canvas is ~1200 px wide.
_AUTO_FONT_AT_1200 = 22
_AUTO_LINE_AT_1200 = 3

_SYSTEM_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


@dataclass(frozen=True)
class AnnotationStyle:
    """Resolved drawing parameters for the *display* canvas size."""

    line_width: int
    font_size: int


@dataclass(frozen=True)
class DisplayTransform:
    """Maps original-image coordinates onto a display-sized canvas."""

    original_width: int
    original_height: int
    display_width: int
    display_height: int
    scale_x: float
    scale_y: float

    @property
    def was_resized(self) -> bool:
        return (
            self.display_width != self.original_width
            or self.display_height != self.original_height
        )


def _round_half_up(value: float) -> int:
    return int(value + 0.5)


def compute_display_transform(
    original_width: int,
    original_height: int,
    *,
    max_width: int = DISPLAY_MAX_WIDTH,
) -> DisplayTransform:
    """
    Compute a letterbox-free downscale that caps width at ``max_width``.

    Images already ≤ ``max_width`` keep their native size (scale = 1).
    """
    orig_w = max(1, int(original_width))
    orig_h = max(1, int(original_height))
    cap = max(1, int(max_width))

    if orig_w <= cap:
        return DisplayTransform(
            original_width=orig_w,
            original_height=orig_h,
            display_width=orig_w,
            display_height=orig_h,
            scale_x=1.0,
            scale_y=1.0,
        )

    scale = cap / float(orig_w)
    disp_w = cap
    disp_h = max(1, _round_half_up(orig_h * scale))
    return DisplayTransform(
        original_width=orig_w,
        original_height=orig_h,
        display_width=disp_w,
        display_height=disp_h,
        scale_x=disp_w / float(orig_w),
        scale_y=disp_h / float(orig_h),
    )


def scale_box_to_display(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    transform: DisplayTransform,
) -> tuple[float, float, float, float]:
    """Map an original-space axis-aligned box onto the display canvas."""
    return (
        x1 * transform.scale_x,
        y1 * transform.scale_y,
        x2 * transform.scale_x,
        y2 * transform.scale_y,
    )


def prepare_display_image(
    image: Image.Image,
    *,
    max_width: int = DISPLAY_MAX_WIDTH,
) -> tuple[Image.Image, DisplayTransform]:
    """Return an RGB display copy (resized if needed) and its transform."""
    rgb = image.convert("RGB")
    transform = compute_display_transform(
        rgb.width,
        rgb.height,
        max_width=max_width,
    )
    if not transform.was_resized:
        return rgb.copy(), transform
    resized = rgb.resize(
        (transform.display_width, transform.display_height),
        Image.Resampling.LANCZOS,
    )
    return resized, transform


def compute_annotation_style(
    width: int,
    height: int,
    scale: AnnotationScale | str = "auto",
) -> AnnotationStyle:
    """
    Derive box thickness and label font size from the *display* canvas size.

    Auto targets ~22 px text / ~3 px stroke on a ~1200 px-wide image, then
    scales mildly with width. Presets multiply that baseline.
    """
    display_w = max(1, int(width))
    _ = height  # kept for API symmetry / future aspect-aware tweaks
    factor = _SCALE_FACTORS.get(str(scale).strip().lower(), 1.0)

    font_size = _round_half_up(display_w * (_AUTO_FONT_AT_1200 / 1200.0) * factor)
    line_width = _round_half_up(display_w * (_AUTO_LINE_AT_1200 / 1200.0) * factor)

    font_size = max(16, min(40, font_size))
    line_width = max(2, min(4, line_width))
    return AnnotationStyle(line_width=line_width, font_size=font_size)


@lru_cache(maxsize=1)
def _font_file_bytes() -> bytes | None:
    """Load bundled TTF bytes once (works in wheels and editable installs)."""
    try:
        resource = resources.files("vision_trainer") / "assets" / "DejaVuSans-Bold.ttf"
        return resource.read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, AttributeError, TypeError, OSError):
        pass
    candidate = Path(__file__).resolve().parent.parent / "assets" / "DejaVuSans-Bold.ttf"
    if candidate.is_file():
        try:
            return candidate.read_bytes()
        except OSError:
            return None
    return None


@lru_cache(maxsize=64)
def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load a scalable TTF (bundled first, then system); fall back to Pillow default."""
    size = max(8, int(size))
    font_bytes = _font_file_bytes()
    if font_bytes:
        try:
            return ImageFont.truetype(io.BytesIO(font_bytes), size=size)
        except OSError:
            pass
    for candidate in _SYSTEM_FONT_CANDIDATES:
        path = Path(candidate)
        if not path.is_file():
            continue
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)  # type: ignore[call-arg]
    except TypeError:
        return ImageFont.load_default()


def format_detection_label(detection: Detection) -> str:
    return f"{detection.class_name} {detection.confidence:.2f}"


def draw_detections(
    image: Image.Image,
    detections: list[Detection],
    *,
    scale: AnnotationScale | str = "auto",
    style: AnnotationStyle | None = None,
    max_display_width: int = DISPLAY_MAX_WIDTH,
    fit_to_display: bool = True,
) -> Image.Image:
    """
    Draw bounding boxes and labels for UI / download.

    When ``fit_to_display`` is True (default), the image is first resized so its
    width does not exceed ``max_display_width``. Detection coordinates are
    assumed to be in *original* image space and are scaled onto that canvas
    before drawing. Font size is computed from the display canvas, so Streamlit
    no longer shrinks huge-resolution annotations into unreadable text.
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

    for detection in detections:
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

        label = format_detection_label(detection)
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
        # Compensate TrueType ascent so glyphs sit inside the filled badge.
        draw.text(
            (text_x + pad_x, text_y + pad_y - text_bbox[1]),
            label,
            fill="white",
            font=font,
        )

    return canvas


def annotated_image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def build_download_filename(original_name: str) -> str:
    stem = Path(original_name).stem or "image"
    return f"prediction_{stem}.jpg"
