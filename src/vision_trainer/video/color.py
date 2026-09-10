"""RGB / BGR conversion helpers for OpenCV ↔ PIL / Ultralytics."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def bgr_to_rgb_array(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR frame to RGB (copy, contiguous)."""
    if frame_bgr is None:
        raise ValueError("Frame BGR vide.")
    array = np.asarray(frame_bgr)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("Frame BGR invalide (attendu HxWx3).")
    # Explicit channel swap — do not rely on cv2 in unit tests.
    return array[:, :, ::-1].copy()


def rgb_to_bgr_array(frame_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB array to OpenCV BGR."""
    array = np.asarray(frame_rgb)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("Frame RGB invalide (attendu HxWx3).")
    return array[:, :, ::-1].copy()


def bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr_to_rgb_array(frame_bgr), mode="RGB")


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return rgb_to_bgr_array(rgb)


def ensure_rgb_pil(image: Image.Image | np.ndarray) -> Image.Image:
    """Accept PIL or RGB numpy array and return an RGB PIL image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("Image RGB invalide.")
    return Image.fromarray(array[:, :, :3].astype(np.uint8), mode="RGB")
