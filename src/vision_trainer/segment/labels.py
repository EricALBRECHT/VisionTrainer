"""Segmentation label parsing and lightweight dataset stats."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PolygonInstance:
    """One YOLO-seg polygon in normalized coordinates."""

    class_id: int
    points: tuple[tuple[float, float], ...]  # normalized [0,1]

    @property
    def num_points(self) -> int:
        return len(self.points)


@dataclass
class SegmentSplitStats:
    name: str
    image_count: int = 0
    instance_count: int = 0
    instances_by_class: dict[str, int] = field(default_factory=dict)
    images_without_labels: list[str] = field(default_factory=list)
    invalid_label_files: int = 0
    total_polygon_points: int = 0
    polygon_count: int = 0

    @property
    def mean_points_per_polygon(self) -> float | None:
        if self.polygon_count <= 0:
            return None
        return self.total_polygon_points / self.polygon_count


@dataclass
class SegmentDatasetInfo:
    """YOLO segmentation dataset (same folder layout as detect, polygon labels)."""

    root: Path
    yaml_path: Path
    class_names: dict[int, str]
    splits: dict[str, Any] = field(default_factory=dict)  # SplitInfo from yolo.models
    split_stats: dict[str, SegmentSplitStats] = field(default_factory=dict)
    detection_label_hits: int = 0

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def total_instances(self) -> int:
        return sum(stats.instance_count for stats in self.split_stats.values())

    def instances_by_class(self) -> dict[str, int]:
        totals: dict[str, int] = {name: 0 for name in self.class_names.values()}
        for stats in self.split_stats.values():
            for name, count in stats.instances_by_class.items():
                totals[name] = totals.get(name, 0) + count
        return totals


def looks_like_detection_bbox_line(parts: list[str]) -> bool:
    """
    True when a label line has exactly class_id + 4 floats (YOLO detect format).

    Segmentation polygons require at least 3 points (6 coordinates).
    """
    if len(parts) != 5:
        return False
    try:
        int(parts[0])
        for value in parts[1:]:
            float(value)
    except ValueError:
        return False
    return True


def parse_segment_label_line(
    line: str,
    *,
    num_classes: int | None = None,
) -> PolygonInstance | None:
    """
    Parse one YOLO-seg label line.

    Returns None for blank lines. Raises ValueError for invalid content.
    """
    stripped = line.strip()
    if not stripped:
        return None
    parts = stripped.split()
    if looks_like_detection_bbox_line(parts):
        raise ValueError(
            "Ce dataset semble contenir des bounding boxes de détection et non "
            "des polygones de segmentation."
        )
    if len(parts) < 7:
        raise ValueError(
            "Polygone insuffisant : au moins class_id + 3 points (x,y) sont requis."
        )
    try:
        class_id = int(parts[0])
    except ValueError as exc:
        raise ValueError(f"Identifiant de classe invalide : {parts[0]!r}") from exc

    coords = parts[1:]
    if len(coords) % 2 != 0:
        raise ValueError("Nombre de coordonnées impair (paires x,y attendues).")
    if len(coords) < 6:
        raise ValueError("Polygone trop court (minimum 3 points).")

    points: list[tuple[float, float]] = []
    values: list[float] = []
    for token in coords:
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ValueError(f"Coordonnée non numérique : {token!r}") from exc

    for index in range(0, len(values), 2):
        x_val, y_val = values[index], values[index + 1]
        if not (0.0 <= x_val <= 1.0 and 0.0 <= y_val <= 1.0):
            raise ValueError(
                f"Coordonnée hors [0, 1] : ({x_val}, {y_val})."
            )
        points.append((x_val, y_val))

    if num_classes is not None and not (0 <= class_id < num_classes):
        raise ValueError(f"Classe {class_id} hors plage [0, {num_classes - 1}].")

    return PolygonInstance(class_id=class_id, points=tuple(points))


def parse_segment_label_file(
    path: Path,
    *,
    class_names: dict[int, str],
) -> tuple[list[PolygonInstance], list[str]]:
    """
    Parse a label file.

    Returns (instances, error_messages). Detection-looking lines yield a
    specific French error and are not converted.
    """
    instances: list[PolygonInstance] = []
    errors: list[str] = []
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [], [f"Impossible de lire {path.name} : {exc}"]

    if not content.strip():
        return [], []

    num_classes = len(class_names)
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            instance = parse_segment_label_line(line, num_classes=None)
        except ValueError as exc:
            errors.append(f"{path.name}:{line_number} : {exc}")
            continue
        if instance is None:
            continue
        if instance.class_id not in class_names:
            errors.append(
                f"{path.name}:{line_number} : classe {instance.class_id} inconnue."
            )
            continue
        # Re-validate range against known ids (may be non-contiguous)
        instances.append(instance)
    return instances, errors


def polygon_to_pixel_points(
    points: tuple[tuple[float, float], ...],
    *,
    image_width: int,
    image_height: int,
) -> list[tuple[float, float]]:
    return [
        (x * image_width, y * image_height)
        for x, y in points
    ]


def polygon_bbox_xyxy(
    pixel_points: list[tuple[float, float]],
) -> tuple[float, float, float, float] | None:
    if not pixel_points:
        return None
    xs = [p[0] for p in pixel_points]
    ys = [p[1] for p in pixel_points]
    return min(xs), min(ys), max(xs), max(ys)
