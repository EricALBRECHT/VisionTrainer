"""Data models for multi-step detection pipelines (format v1 + v2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from vision_trainer.classify.reject import ClassScore
from vision_trainer.inference.models import Detection

# Current on-disk format when saving. Readers still accept format_version 1.
PIPELINE_FORMAT_VERSION = 2
PIPELINE_FORMAT_VERSION_V1 = 1

RefineStatus = Literal["ok", "unknown", "uncertain", "skipped", "error"]


@dataclass
class ClassificationStage:
    """Optional classify step for one detected class."""

    enabled: bool = False
    run_id: str | None = None
    confidence_threshold: float = 0.80
    margin_threshold: float = 0.10
    top_n: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "run_id": self.run_id,
            "confidence_threshold": float(self.confidence_threshold),
            "margin_threshold": float(self.margin_threshold),
            "top_n": int(self.top_n),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ClassificationStage:
        raw = data if isinstance(data, dict) else {}
        run_id = raw.get("run_id", raw.get("classifier_run_id"))
        return cls(
            enabled=bool(raw.get("enabled", False)),
            run_id=str(run_id) if run_id not in (None, "") else None,
            confidence_threshold=float(raw.get("confidence_threshold", 0.80)),
            margin_threshold=float(raw.get("margin_threshold", 0.10)),
            top_n=max(1, int(raw.get("top_n", 3))),
        )


@dataclass
class SegmentationStage:
    """Optional segment step for one detected class (runs on the same crop)."""

    enabled: bool = False
    run_id: str | None = None
    confidence_threshold: float = 0.25
    crop_padding: float | None = None  # None → pipeline default

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "run_id": self.run_id,
            "confidence_threshold": float(self.confidence_threshold),
            "crop_padding": (
                None if self.crop_padding is None else float(self.crop_padding)
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SegmentationStage:
        raw = data if isinstance(data, dict) else {}
        run_id = raw.get("run_id", raw.get("segmenter_run_id"))
        padding = raw.get("crop_padding", None)
        return cls(
            enabled=bool(raw.get("enabled", False)),
            run_id=str(run_id) if run_id not in (None, "") else None,
            confidence_threshold=float(raw.get("confidence_threshold", 0.25)),
            crop_padding=None if padding is None else float(padding),
        )


@dataclass
class ClassMapping:
    """
    Per-detected-class refinements.

    Accepts legacy v1 kwargs (``enabled``, ``classifier_run_id``, …) for
    compatibility with existing tests and callers.
    """

    classification: ClassificationStage = field(default_factory=ClassificationStage)
    segmentation: SegmentationStage = field(default_factory=SegmentationStage)

    def __init__(
        self,
        classification: ClassificationStage | None = None,
        segmentation: SegmentationStage | None = None,
        *,
        enabled: bool | None = None,
        classifier_run_id: str | None = None,
        confidence_threshold: float | None = None,
        margin_threshold: float | None = None,
        top_n: int | None = None,
        segment_enabled: bool | None = None,
        segmenter_run_id: str | None = None,
        segment_confidence_threshold: float | None = None,
        segment_crop_padding: float | None = None,
    ) -> None:
        if classification is not None:
            self.classification = classification
        elif any(
            value is not None
            for value in (
                enabled,
                classifier_run_id,
                confidence_threshold,
                margin_threshold,
                top_n,
            )
        ):
            self.classification = ClassificationStage(
                enabled=bool(enabled) if enabled is not None else bool(classifier_run_id),
                run_id=classifier_run_id,
                confidence_threshold=(
                    0.80 if confidence_threshold is None else float(confidence_threshold)
                ),
                margin_threshold=(
                    0.10 if margin_threshold is None else float(margin_threshold)
                ),
                top_n=3 if top_n is None else max(1, int(top_n)),
            )
        else:
            self.classification = ClassificationStage()

        if segmentation is not None:
            self.segmentation = segmentation
        elif any(
            value is not None
            for value in (
                segment_enabled,
                segmenter_run_id,
                segment_confidence_threshold,
                segment_crop_padding,
            )
        ):
            self.segmentation = SegmentationStage(
                enabled=(
                    bool(segment_enabled)
                    if segment_enabled is not None
                    else bool(segmenter_run_id)
                ),
                run_id=segmenter_run_id,
                confidence_threshold=(
                    0.25
                    if segment_confidence_threshold is None
                    else float(segment_confidence_threshold)
                ),
                crop_padding=segment_crop_padding,
            )
        else:
            self.segmentation = SegmentationStage()

    @property
    def enabled(self) -> bool:
        """True when any secondary stage is enabled (v1-compatible)."""
        return self.classification.enabled or self.segmentation.enabled

    @property
    def classifier_run_id(self) -> str | None:
        return self.classification.run_id

    @property
    def confidence_threshold(self) -> float:
        return self.classification.confidence_threshold

    @property
    def margin_threshold(self) -> float:
        return self.classification.margin_threshold

    @property
    def top_n(self) -> int:
        return self.classification.top_n

    def to_dict(self) -> dict[str, Any]:
        """Serialize as format v2 nested stages."""
        return {
            "classification": self.classification.to_dict(),
            "segmentation": self.segmentation.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ClassMapping:
        raw = data if isinstance(data, dict) else {}
        # v2 nested
        if "classification" in raw or "segmentation" in raw:
            return cls(
                classification=ClassificationStage.from_dict(
                    raw.get("classification") if isinstance(raw.get("classification"), dict) else {}
                ),
                segmentation=SegmentationStage.from_dict(
                    raw.get("segmentation") if isinstance(raw.get("segmentation"), dict) else {}
                ),
            )
        # v1 flat
        return cls(
            enabled=bool(raw.get("enabled", False)),
            classifier_run_id=(
                str(raw["classifier_run_id"])
                if raw.get("classifier_run_id") not in (None, "")
                else None
            ),
            confidence_threshold=float(raw.get("confidence_threshold", 0.80)),
            margin_threshold=float(raw.get("margin_threshold", 0.10)),
            top_n=max(1, int(raw.get("top_n", 3))),
        )


@dataclass
class PipelineConfig:
    """Saved pipeline: one detector + optional classify/segment per class."""

    pipeline_id: str
    name: str
    detector_run_id: str
    mappings: dict[str, ClassMapping] = field(default_factory=dict)
    crop_padding: float = 0.05
    detect_conf: float = 0.25
    detect_iou: float = 0.45
    format_version: int = PIPELINE_FORMAT_VERSION
    created_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": int(PIPELINE_FORMAT_VERSION),
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "detector_run_id": self.detector_run_id,
            "crop_padding": float(self.crop_padding),
            "detect_conf": float(self.detect_conf),
            "detect_iou": float(self.detect_iou),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "mappings": {key: value.to_dict() for key, value in self.mappings.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        if not isinstance(data, dict):
            raise ValueError("Pipeline invalide : objet JSON attendu.")
        pipeline_id = str(data.get("pipeline_id") or "").strip()
        name = str(data.get("name") or "").strip()
        detector_run_id = str(data.get("detector_run_id") or "").strip()
        if not pipeline_id:
            raise ValueError("Pipeline invalide : pipeline_id manquant.")
        if not name:
            raise ValueError("Pipeline invalide : name manquant.")
        if not detector_run_id:
            raise ValueError("Pipeline invalide : detector_run_id manquant.")

        raw_mappings = data.get("mappings") or {}
        if not isinstance(raw_mappings, dict):
            raise ValueError("Pipeline invalide : mappings doit être un objet.")

        mappings = {
            str(class_name): ClassMapping.from_dict(
                payload if isinstance(payload, dict) else {}
            )
            for class_name, payload in raw_mappings.items()
        }
        # Keep original format_version from file (in-memory stages are always normalized).
        file_version = int(data.get("format_version") or PIPELINE_FORMAT_VERSION_V1)
        return cls(
            pipeline_id=pipeline_id,
            name=name,
            detector_run_id=detector_run_id,
            mappings=mappings,
            crop_padding=float(data.get("crop_padding", 0.05)),
            detect_conf=float(data.get("detect_conf", 0.25)),
            detect_iou=float(data.get("detect_iou", 0.45)),
            format_version=file_version,
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass(frozen=True)
class ClassificationRefinement:
    status: RefineStatus
    class_name: str | None = None
    confidence: float | None = None
    top_n: tuple[ClassScore, ...] = ()
    reason: str | None = None
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "top_n": [
                {
                    "class_id": item.class_id,
                    "class_name": item.class_name,
                    "confidence": item.confidence,
                }
                for item in self.top_n
            ],
            "reason": self.reason,
            "warning": self.warning,
        }


@dataclass(frozen=True)
class SegmentationInstanceResult:
    class_id: int
    class_name: str
    confidence: float
    polygon_global: tuple[tuple[float, float], ...] = ()
    bbox_global: tuple[float, float, float, float] | None = None
    mask_area_pixels: int = 0
    mask_area_ratio_crop: float = 0.0
    mask_area_ratio_image: float = 0.0
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "polygon_global": [[float(x), float(y)] for x, y in self.polygon_global],
            "bbox_global": list(self.bbox_global) if self.bbox_global else None,
            "mask_area_pixels": self.mask_area_pixels,
            "mask_area_ratio_crop": self.mask_area_ratio_crop,
            "mask_area_ratio_image": self.mask_area_ratio_image,
            "warning": self.warning,
        }


@dataclass(frozen=True)
class SegmentationRefinement:
    """Outcome of the optional segment stage (may contain multiple masks)."""

    status: RefineStatus
    instances: tuple[SegmentationInstanceResult, ...] = ()
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "instances": [item.to_dict() for item in self.instances],
            "warning": self.warning,
        }


@dataclass
class PipelineTimings:
    detection_ms: float = 0.0
    classification_ms: float = 0.0
    segmentation_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_ms": round(self.detection_ms, 3),
            "classification_ms": round(self.classification_ms, 3),
            "segmentation_ms": round(self.segmentation_ms, 3),
            "total_ms": round(self.total_ms, 3),
        }


@dataclass(frozen=True)
class EnrichedDetection:
    detection: Detection
    refined: bool
    classification: ClassificationRefinement | None = None
    segmentation: SegmentationRefinement | None = None
    crop_box: tuple[int, int, int, int] | None = None

    @property
    def segmentations(self) -> tuple[SegmentationInstanceResult, ...]:
        if self.segmentation is None:
            return ()
        return self.segmentation.instances

    @property
    def display_label(self) -> str:
        """Short label for overlay rendering."""
        det = self.detection
        base = f"{det.class_name} {det.confidence:.2f}"
        if not self.refined:
            return base

        parts = [base]
        if self.classification is not None:
            refinement = self.classification
            if refinement.status == "ok" and refinement.class_name:
                conf = refinement.confidence if refinement.confidence is not None else 0.0
                parts = [f"{det.class_name} {det.confidence:.2f} → {refinement.class_name} {conf:.2f}"]
            elif refinement.status == "unknown":
                parts = [f"{det.class_name} → INCONNU"]
            elif refinement.status == "uncertain":
                parts = [f"{det.class_name} → INCERTAIN"]
            elif refinement.status == "error":
                parts = [base]

        if self.segmentation is not None and self.segmentation.status == "ok":
            masks = self.segmentation.instances
            if len(masks) == 1:
                m0 = masks[0]
                parts.append(f"+ {m0.class_name} {m0.confidence:.2f}")
            elif len(masks) > 1:
                parts.append(f"+ {len(masks)} masques")
        return " ".join(parts) if len(parts) > 1 else parts[0]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "detected_class": self.detection.class_name,
            "detected_class_id": self.detection.class_id,
            "detected_confidence": self.detection.confidence,
            "bbox": [
                self.detection.x1,
                self.detection.y1,
                self.detection.x2,
                self.detection.y2,
            ],
            "refined": self.refined,
            "crop_box": list(self.crop_box) if self.crop_box else None,
        }
        if self.classification is not None:
            payload["classification"] = self.classification.to_dict()
        if self.segmentation is not None:
            payload["segmentation"] = self.segmentation.to_dict()
            payload["segmentations"] = [
                item.to_dict() for item in self.segmentation.instances
            ]
        return payload


@dataclass
class PipelineResult:
    items: list[EnrichedDetection] = field(default_factory=list)
    detector_class_names: dict[int, str] = field(default_factory=dict)
    image_width: int = 0
    image_height: int = 0
    warnings: list[str] = field(default_factory=list)
    pipeline_id: str | None = None
    pipeline_name: str | None = None
    timings: PipelineTimings = field(default_factory=PipelineTimings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "warnings": list(self.warnings),
            "timings": self.timings.to_dict(),
            "items": [item.to_dict() for item in self.items],
            "detector_class_names": {
                str(k): v for k, v in self.detector_class_names.items()
            },
        }

    def to_json_dict(self) -> dict[str, Any]:
        """Export-friendly payload (no bitmap masks)."""
        return self.to_dict()
