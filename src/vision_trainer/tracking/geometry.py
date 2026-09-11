"""Geometry helpers for tracking."""

from __future__ import annotations


def bbox_center(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    """Return (center_x, center_y) of an axis-aligned box."""
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)


def frame_stride_tracking_warning(frame_stride: int) -> str | None:
    """Warn when stride > 1 may hurt track continuity."""
    if int(frame_stride) > 1:
        return (
            "Le saut de frames (stride > 1) peut réduire la stabilité du tracking : "
            "le tracker reçoit moins d'observations temporelles."
        )
    return None
