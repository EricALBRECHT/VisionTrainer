"""Persistent JSON storage for pipelines under data/pipelines/."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vision_trainer.io_utils import atomic_write_json
from vision_trainer.paths import get_pipelines_dir, get_run_dir, get_runs_dir
from vision_trainer.pipeline.models import (
    PIPELINE_FORMAT_VERSION,
    ClassMapping,
    PipelineConfig,
)
from vision_trainer.tasks import KNOWN_TASKS, normalize_task

PIPELINE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")


class PipelineStoreError(ValueError):
    """Raised when a pipeline cannot be loaded or saved."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_pipeline_filename(pipeline_id: str) -> str:
    if not PIPELINE_ID_PATTERN.fullmatch(pipeline_id):
        raise PipelineStoreError(
            "Identifiant de pipeline invalide (lettres, chiffres, ._- uniquement)."
        )
    return f"{pipeline_id}.json"


def pipeline_path(pipeline_id: str, *, root: Path | None = None) -> Path:
    base = root if root is not None else get_pipelines_dir()
    return base / _safe_pipeline_filename(pipeline_id)


def resolve_run_weights(run_id: str, *, prefer_best: bool = True) -> Path:
    """Resolve weights for a training run (best.pt preferred, else last.pt)."""
    run_dir = get_run_dir(run_id)
    if not run_dir.is_dir():
        raise PipelineStoreError(f"Run introuvable : {run_id}")
    weights_dir = run_dir / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    if prefer_best and best.is_file():
        return best
    if last.is_file():
        return last
    if best.is_file():
        return best
    raise PipelineStoreError(f"Poids introuvables pour le run {run_id}")


def read_run_task(run_id: str) -> str:
    """Return task from run config/status (default detect for legacy runs)."""
    run_dir = get_run_dir(run_id)
    for filename in ("config.json", "status.json", "request.json"):
        config_path = run_dir / filename
        if not config_path.is_file():
            continue
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        task = str(payload.get("task") or "").strip().lower()
        if task in KNOWN_TASKS:
            return normalize_task(task)
    return "detect"


def list_pipelines(*, root: Path | None = None) -> list[PipelineConfig]:
    """List valid pipelines; skip unreadable files without crashing."""
    base = root if root is not None else get_pipelines_dir()
    if not base.is_dir():
        return []
    items: list[PipelineConfig] = []
    for path in sorted(base.glob("*.json")):
        try:
            items.append(load_pipeline(path.stem, root=base))
        except PipelineStoreError:
            continue
    items.sort(key=lambda p: (p.name.lower(), p.pipeline_id))
    return items


def load_pipeline(pipeline_id: str, *, root: Path | None = None) -> PipelineConfig:
    """
    Load a pipeline JSON.

    Format v1 files are normalized in memory to nested classification stages;
    the on-disk file is not rewritten.
    """
    path = pipeline_path(pipeline_id, root=root)
    if not path.is_file():
        raise PipelineStoreError(f"Pipeline introuvable : {pipeline_id}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineStoreError(f"Pipeline JSON invalide ({path.name}) : {exc}") from exc
    except OSError as exc:
        raise PipelineStoreError(f"Impossible de lire le pipeline : {exc}") from exc
    try:
        return PipelineConfig.from_dict(payload)
    except ValueError as exc:
        raise PipelineStoreError(str(exc)) from exc


def save_pipeline(config: PipelineConfig, *, root: Path | None = None) -> Path:
    """Validate lightly and persist as format v2 (explicit user save migrates)."""
    validate_pipeline_config(config, check_weights=False)
    now = _utc_now_iso()
    if not config.created_at:
        config.created_at = now
    config.updated_at = now
    config.format_version = PIPELINE_FORMAT_VERSION
    path = pipeline_path(config.pipeline_id, root=root)
    atomic_write_json(path, config.to_dict())
    return path


def delete_pipeline(pipeline_id: str, *, root: Path | None = None) -> None:
    path = pipeline_path(pipeline_id, root=root)
    if not path.is_file():
        raise PipelineStoreError(f"Pipeline introuvable : {pipeline_id}")
    path.unlink()


def duplicate_pipeline(
    pipeline_id: str,
    *,
    new_id: str,
    new_name: str,
    root: Path | None = None,
) -> PipelineConfig:
    source = load_pipeline(pipeline_id, root=root)
    clone = PipelineConfig.from_dict(source.to_dict())
    clone.pipeline_id = new_id
    clone.name = new_name
    clone.created_at = None
    clone.updated_at = None
    save_pipeline(clone, root=root)
    return clone


def validate_pipeline_config(
    config: PipelineConfig,
    *,
    check_weights: bool = True,
) -> list[str]:
    """Validate config structure and optionally referenced weights."""
    warnings: list[str] = []
    if not config.name.strip():
        raise PipelineStoreError("Le nom du pipeline est obligatoire.")
    if not PIPELINE_ID_PATTERN.fullmatch(config.pipeline_id):
        raise PipelineStoreError(
            "Identifiant de pipeline invalide (lettres, chiffres, ._- uniquement)."
        )
    if not 0.0 <= config.crop_padding <= 0.5:
        raise PipelineStoreError("crop_padding doit être entre 0 et 0.5.")
    if not 0.05 <= config.detect_conf <= 0.95:
        raise PipelineStoreError("detect_conf doit être entre 0.05 et 0.95.")
    if not 0.05 <= config.detect_iou <= 0.95:
        raise PipelineStoreError("detect_iou doit être entre 0.05 et 0.95.")

    for class_name, mapping in config.mappings.items():
        if not class_name.strip():
            raise PipelineStoreError("Une classe de mapping a un nom vide.")

        cls_stage = mapping.classification
        seg_stage = mapping.segmentation

        if cls_stage.enabled:
            if not cls_stage.run_id:
                raise PipelineStoreError(
                    f"Classe « {class_name} » : classificateur requis quand "
                    "la classification est active."
                )
            if not 0.0 <= cls_stage.confidence_threshold <= 1.0:
                raise PipelineStoreError(
                    f"Classe « {class_name} » : seuil de confiance classification invalide."
                )
            if not 0.0 <= cls_stage.margin_threshold <= 1.0:
                raise PipelineStoreError(
                    f"Classe « {class_name} » : écart Top1/Top2 invalide."
                )
            if cls_stage.top_n < 1:
                raise PipelineStoreError(f"Classe « {class_name} » : top_n invalide.")

        if seg_stage.enabled:
            if not seg_stage.run_id:
                raise PipelineStoreError(
                    f"Classe « {class_name} » : modèle segmentation requis quand "
                    "la segmentation est active."
                )
            if not 0.0 <= seg_stage.confidence_threshold <= 1.0:
                raise PipelineStoreError(
                    f"Classe « {class_name} » : seuil de confiance segmentation invalide."
                )
            if seg_stage.crop_padding is not None and not (
                0.0 <= seg_stage.crop_padding <= 0.5
            ):
                raise PipelineStoreError(
                    f"Classe « {class_name} » : crop_padding segmentation invalide."
                )

    if check_weights:
        try:
            resolve_run_weights(config.detector_run_id)
        except PipelineStoreError as exc:
            raise PipelineStoreError(f"Détecteur : {exc}") from exc
        if read_run_task(config.detector_run_id) != "detect":
            raise PipelineStoreError(
                f"Le run détecteur « {config.detector_run_id} » n'est pas un modèle de détection."
            )

        for class_name, mapping in config.mappings.items():
            cls_stage = mapping.classification
            if cls_stage.enabled and cls_stage.run_id:
                try:
                    resolve_run_weights(cls_stage.run_id)
                except PipelineStoreError as exc:
                    raise PipelineStoreError(
                        f"Classificateur pour « {class_name} » : {exc}"
                    ) from exc
                if read_run_task(cls_stage.run_id) != "classify":
                    raise PipelineStoreError(
                        f"Le run « {cls_stage.run_id} » associé à « {class_name} » "
                        "n'est pas un classificateur."
                    )

            seg_stage = mapping.segmentation
            if seg_stage.enabled and seg_stage.run_id:
                try:
                    resolve_run_weights(seg_stage.run_id)
                except PipelineStoreError as exc:
                    raise PipelineStoreError(
                        f"Segmenter pour « {class_name} » : {exc}"
                    ) from exc
                if read_run_task(seg_stage.run_id) != "segment":
                    raise PipelineStoreError(
                        f"Le run « {seg_stage.run_id} » associé à « {class_name} » "
                        "n'est pas un modèle de segmentation."
                    )

    return warnings


def make_pipeline_id(name: str) -> str:
    """Build a stable-ish id from name + timestamp."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip())[:40].strip("-._") or "pipeline"
    return f"{stamp}-{slug}"


def _list_run_ids_for_task(task: str) -> list[str]:
    runs_dir = get_runs_dir()
    if not runs_dir.is_dir():
        return []
    ids: list[str] = []
    for path in sorted(runs_dir.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        try:
            if read_run_task(path.name) != task:
                continue
            resolve_run_weights(path.name)
        except PipelineStoreError:
            continue
        ids.append(path.name)
    return ids


def list_detect_run_ids() -> list[str]:
    return _list_run_ids_for_task("detect")


def list_classify_run_ids() -> list[str]:
    return _list_run_ids_for_task("classify")


def list_segment_run_ids() -> list[str]:
    return _list_run_ids_for_task("segment")


def mapping_summary(config: PipelineConfig) -> list[dict[str, Any]]:
    """UI-friendly summary of mappings."""
    rows: list[dict[str, Any]] = []
    for class_name, mapping in sorted(config.mappings.items()):
        rows.append(
            {
                "class_name": class_name,
                "enabled": mapping.enabled,
                "classification_enabled": mapping.classification.enabled,
                "classifier_run_id": mapping.classification.run_id,
                "confidence_threshold": mapping.classification.confidence_threshold,
                "margin_threshold": mapping.classification.margin_threshold,
                "top_n": mapping.classification.top_n,
                "segmentation_enabled": mapping.segmentation.enabled,
                "segmenter_run_id": mapping.segmentation.run_id,
                "segment_confidence_threshold": mapping.segmentation.confidence_threshold,
                "segment_crop_padding": mapping.segmentation.crop_padding,
            }
        )
    return rows
