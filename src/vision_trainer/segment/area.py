"""Mask area utilities (pixel counts — not physical units)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw


def mask_area_pixels(mask: np.ndarray | Sequence[Sequence[float]]) -> int:
    """
    Count positive pixels in a 2D mask.

    Accepts bool / int / float arrays; values > 0 count as foreground.
    """
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("Le masque doit être un tableau 2D.")
    return int(np.count_nonzero(array > 0))


def mask_area_ratio(
    area_pixels: int,
    *,
    image_width: int,
    image_height: int,
) -> float:
    """Return area_pixels / (width * height). Not a physical surface."""
    if image_width <= 0 or image_height <= 0:
        return 0.0
    total = int(image_width) * int(image_height)
    if total <= 0:
        return 0.0
    return max(0.0, float(area_pixels) / float(total))


def polygon_area_pixels(
    polygon_xy: Sequence[tuple[float, float]],
    *,
    image_width: int,
    image_height: int,
) -> int:
    """Rasterize a polygon onto an image-sized canvas and count pixels."""
    if image_width <= 0 or image_height <= 0 or len(polygon_xy) < 3:
        return 0
    canvas = Image.new("1", (int(image_width), int(image_height)), 0)
    draw = ImageDraw.Draw(canvas)
    # Clamp points lightly into drawable space
    points = [
        (
            max(0, min(image_width - 1, float(x))),
            max(0, min(image_height - 1, float(y))),
        )
        for x, y in polygon_xy
    ]
    draw.polygon(points, outline=1, fill=1)
    try:
        data = canvas.get_flattened_data()  # type: ignore[attr-defined]
    except AttributeError:
        data = canvas.getdata()
    return int(sum(data))
