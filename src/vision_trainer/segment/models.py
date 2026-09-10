"""Structured segmentation inference results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SegmentInstance:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    # Polygon in original image pixel coordinates (may be empty if mask missing).
    polygon: tuple[tuple[float, float], ...] = ()
    mask_area_pixels: int = 0
    mask_area_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "polygon": [[float(x), float(y)] for x, y in self.polygon],
            "mask_area_pixels": self.mask_area_pixels,
            "mask_area_ratio": self.mask_area_ratio,
        }


@dataclass
class SegmentationResult:
    instances: list[SegmentInstance] = field(default_factory=list)
    class_names: dict[int, str] = field(default_factory=dict)
    image_width: int = 0
    image_height: int = 0
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "device": self.device,
            "class_names": {str(k): v for k, v in self.class_names.items()},
            "instances": [item.to_dict() for item in self.instances],
        }

    def to_json_dict(self) -> dict[str, Any]:
        """Export-friendly payload (polygons, no full bitmap masks)."""
        return self.to_dict()
