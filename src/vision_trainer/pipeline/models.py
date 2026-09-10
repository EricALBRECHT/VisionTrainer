"""Data models for detection → classification pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from vision_trainer.classify.reject import ClassScore
from vision_trainer.inference.models import Detection

PIPELINE_FORMAT_VERSION = 1

RefineStatus = Literal["ok", "unknown", "uncertain", "skipped", "error"]


@dataclass
class ClassMapping:
    """Per-detected-class refinement settings."""

    enabled: bool = False
    classifier_run_id: str | None = None
    confidence_threshold: float = 0.80
    margin_threshold: float = 0.10
    top_n: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "classifier_run_id": self.classifier_run_id,
            "confidence_threshold": float(self.confidence_threshold),
            "margin_threshold": float(self.margin_threshold),
            "top_n": int(self.top_n),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ClassMapping:
        raw = data if isinstance(data, dict) else {}
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
    """Saved pipeline: one detector + optional classifiers per class."""

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
            "format_version": int(self.format_version),
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
            str(class_name): ClassMapping.from_dict(payload if isinstance(payload, dict) else {})
            for class_name, payload in raw_mappings.items()
        }
        return cls(
            pipeline_id=pipeline_id,
            name=name,
            detector_run_id=detector_run_id,
            mappings=mappings,
            crop_padding=float(data.get("crop_padding", 0.05)),
            detect_conf=float(data.get("detect_conf", 0.25)),
            detect_iou=float(data.get("detect_iou", 0.45)),
            format_version=int(data.get("format_version") or PIPELINE_FORMAT_VERSION),
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
class EnrichedDetection:
    detection: Detection
    refined: bool
    classification: ClassificationRefinement | None = None
    crop_box: tuple[int, int, int, int] | None = None

    @property
    def display_label(self) -> str:
        """Short label for overlay rendering."""
        det = self.detection
        det_pct = int(round(det.confidence * 100))
        if not self.refined or self.classification is None:
            return f"{det.class_name} {det.confidence:.2f}"

        refinement = self.classification
        if refinement.status == "ok" and refinement.class_name:
            conf = refinement.confidence if refinement.confidence is not None else 0.0
            return f"{det.class_name} → {refinement.class_name} {conf:.2f}"
        if refinement.status == "unknown":
            return f"{det.class_name} → INCONNU"
        if refinement.status == "uncertain":
            return f"{det.class_name} → INCERTAIN"
        if refinement.status == "error":
            return f"{det.class_name} {det.confidence:.2f}"
        return f"{det.class_name} {det.confidence:.2f}"

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "warnings": list(self.warnings),
            "items": [item.to_dict() for item in self.items],
            "detector_class_names": {
                str(k): v for k, v in self.detector_class_names.items()
            },
        }
