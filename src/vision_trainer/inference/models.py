from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class AvailableModel:
    run_id: str
    weights_path: str
    label: str


@dataclass
class InferenceResult:
    detections: list[Detection] = field(default_factory=list)
    class_names: dict[int, str] = field(default_factory=dict)
    image_width: int = 0
    image_height: int = 0

    @property
    def count(self) -> int:
        return len(self.detections)
