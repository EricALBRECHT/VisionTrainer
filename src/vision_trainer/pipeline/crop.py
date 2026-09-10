"""Crop extraction and local→global coordinate transforms."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from PIL import Image


class CropError(ValueError):
    """Raised when a detection box cannot yield a usable crop."""


def clamp_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Clamp and order box coordinates to image bounds."""
    if image_width <= 0 or image_height <= 0:
        raise CropError("Dimensions d'image invalides pour le crop.")

    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)

    left = max(0.0, min(float(image_width), left))
    right = max(0.0, min(float(image_width), right))
    top = max(0.0, min(float(image_height), top))
    bottom = max(0.0, min(float(image_height), bottom))

    if right <= left or bottom <= top:
        raise CropError("Boîte de détection vide ou hors image.")
    return left, top, right, bottom


def padded_crop_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    image_width: int,
    image_height: int,
    padding: float = 0.05,
) -> tuple[int, int, int, int]:
    """
    Compute integer crop box with relative padding around the detection.

    ``padding`` is a fraction of the box width/height added on each side,
    then clamped to the image. Returns (left, top, right, bottom) inclusive-
    exclusive PIL crop coordinates.
    """
    if padding < 0:
        raise CropError("Le padding de crop ne peut pas être négatif.")
    if padding > 1.0:
        raise CropError("Le padding de crop est trop élevé (max 1.0).")

    left, top, right, bottom = clamp_box(
        x1,
        y1,
        x2,
        y2,
        image_width=image_width,
        image_height=image_height,
    )
    box_w = right - left
    box_h = bottom - top
    pad_x = box_w * float(padding)
    pad_y = box_h * float(padding)

    crop_left = int(math.floor(left - pad_x))
    crop_top = int(math.floor(top - pad_y))
    crop_right = int(math.ceil(right + pad_x))
    crop_bottom = int(math.ceil(bottom + pad_y))

    crop_left = max(0, min(image_width, crop_left))
    crop_right = max(0, min(image_width, crop_right))
    crop_top = max(0, min(image_height, crop_top))
    crop_bottom = max(0, min(image_height, crop_bottom))

    if crop_right <= crop_left or crop_bottom <= crop_top:
        raise CropError("Crop résultant vide après padding / clamp.")
    return crop_left, crop_top, crop_right, crop_bottom


@dataclass(frozen=True)
class CropRegion:
    """
    Actual cropped region in original-image pixel space.

    Secondary models (classify / segment) run on ``image``; their local
    coordinates must be remapped with the helpers below.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    image: Image.Image

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def local_to_global_point(self, x: float, y: float) -> tuple[float, float]:
        return float(self.x1) + float(x), float(self.y1) + float(y)

    def local_to_global_polygon(
        self,
        points: Sequence[tuple[float, float]],
    ) -> tuple[tuple[float, float], ...]:
        return tuple(self.local_to_global_point(x, y) for x, y in points)

    def local_to_global_bbox(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> tuple[float, float, float, float]:
        gx1, gy1 = self.local_to_global_point(x1, y1)
        gx2, gy2 = self.local_to_global_point(x2, y2)
        return min(gx1, gx2), min(gy1, gy2), max(gx1, gx2), max(gy1, gy2)


def crop_region_from_detection(
    image: Image.Image,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    padding: float = 0.05,
) -> CropRegion:
    """Build a CropRegion from the original image (not a display canvas)."""
    crop, box = crop_from_detection(image, x1, y1, x2, y2, padding=padding)
    return CropRegion(x1=box[0], y1=box[1], x2=box[2], y2=box[3], image=crop)


def crop_from_detection(
    image: Image.Image,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    padding: float = 0.05,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """
    Crop from the original image (not a display-resized copy).

    Returns (crop_rgb, (left, top, right, bottom)).
    """
    if not isinstance(image, Image.Image):
        raise CropError("Image PIL attendue pour le crop.")
    rgb = image.convert("RGB")
    width, height = rgb.size
    box = padded_crop_box(
        x1,
        y1,
        x2,
        y2,
        image_width=width,
        image_height=height,
        padding=padding,
    )
    crop = rgb.crop(box)
    if crop.size[0] < 1 or crop.size[1] < 1:
        raise CropError("Crop résultant invalide.")
    return crop, box
