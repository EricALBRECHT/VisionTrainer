"""Crop extraction from original images for pipeline refinement."""

from __future__ import annotations

import math
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
