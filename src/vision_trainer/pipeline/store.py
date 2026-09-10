"""Persistent JSON storage for pipelines under data/pipelines/."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vision_trainer.io_utils import atomic_write_json
from vision_trainer.paths import get_pipelines_dir, get_run_dir, get_runs_dir
from vision_trainer.pipeline.models import ClassMapping, PipelineConfig

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
    """Return task from run config.json (default detect for legacy runs)."""
    config_path = get_run_dir(run_id) / "config.json"
    if not config_path.is_file():
        return "detect"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "detect"
    task = str(payload.get("task") or "detect").strip().lower()
    return task if task in {"detect", "classify"} else "detect"


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
    """Validate lightly and persist the pipeline config."""
    validate_pipeline_config(config, check_weights=False)
    now = _utc_now_iso()
    if not config.created_at:
        config.created_at = now
    config.updated_at = now
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
    """
    Validate config structure and optionally referenced weights.

    Returns non-fatal warnings (e.g. class mapping for unknown detector class
    is checked only when detector names are supplied separately).
    """
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
        if mapping.enabled:
            if not mapping.classifier_run_id:
                raise PipelineStoreError(
                    f"Classe « {class_name} » : classificateur requis quand l'affinement est actif."
                )
            if not 0.0 <= mapping.confidence_threshold <= 1.0:
                raise PipelineStoreError(
                    f"Classe « {class_name} » : seuil de confiance invalide."
                )
            if not 0.0 <= mapping.margin_threshold <= 1.0:
                raise PipelineStoreError(
                    f"Classe « {class_name} » : écart Top1/Top2 invalide."
                )
            if mapping.top_n < 1:
                raise PipelineStoreError(f"Classe « {class_name} » : top_n invalide.")

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
            if not mapping.enabled or not mapping.classifier_run_id:
                continue
            try:
                resolve_run_weights(mapping.classifier_run_id)
            except PipelineStoreError as exc:
                raise PipelineStoreError(
                    f"Classificateur pour « {class_name} » : {exc}"
                ) from exc
            if read_run_task(mapping.classifier_run_id) != "classify":
                raise PipelineStoreError(
                    f"Le run « {mapping.classifier_run_id} » associé à « {class_name} » "
                    "n'est pas un classificateur."
                )

    return warnings


def make_pipeline_id(name: str) -> str:
    """Build a stable-ish id from name + timestamp."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip())[:40].strip("-._") or "pipeline"
    return f"{stamp}-{slug}"


def list_detect_run_ids() -> list[str]:
    """Run ids under runs/ that look like detectors with weights."""
    runs_dir = get_runs_dir()
    if not runs_dir.is_dir():
        return []
    ids: list[str] = []
    for path in sorted(runs_dir.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        try:
            if read_run_task(path.name) != "detect":
                continue
            resolve_run_weights(path.name)
        except PipelineStoreError:
            continue
        ids.append(path.name)
    return ids


def list_classify_run_ids() -> list[str]:
    runs_dir = get_runs_dir()
    if not runs_dir.is_dir():
        return []
    ids: list[str] = []
    for path in sorted(runs_dir.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        try:
            if read_run_task(path.name) != "classify":
                continue
            resolve_run_weights(path.name)
        except PipelineStoreError:
            continue
        ids.append(path.name)
    return ids


def mapping_summary(config: PipelineConfig) -> list[dict[str, Any]]:
    """UI-friendly summary of active mappings."""
    rows: list[dict[str, Any]] = []
    for class_name, mapping in sorted(config.mappings.items()):
        rows.append(
            {
                "class_name": class_name,
                "enabled": mapping.enabled,
                "classifier_run_id": mapping.classifier_run_id,
                "confidence_threshold": mapping.confidence_threshold,
                "margin_threshold": mapping.margin_threshold,
                "top_n": mapping.top_n,
            }
        )
    return rows
