"""External / mounted datasets (no ZIP upload, no full copy).

Datasets live under a dedicated root (default ``/datasets``), typically a
Docker bind mount in read-only mode. VisionTrainer may only write small
metadata under the internal data root.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from vision_trainer.io_utils import atomic_write_json
from vision_trainer.paths import get_datasets_dir, get_external_datasets_root

SourceType = Literal["uploaded", "external"]

SOURCE_UPLOADED: SourceType = "uploaded"
SOURCE_EXTERNAL: SourceType = "external"

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ExternalDatasetError(ValueError):
    """Invalid external dataset path or missing mount."""


def normalize_source_type(value: object | None) -> SourceType:
    """Legacy payloads without ``source_type`` are treated as uploaded."""
    if value is None or value == "":
        return SOURCE_UPLOADED
    text = str(value).strip().lower()
    if text == SOURCE_EXTERNAL:
        return SOURCE_EXTERNAL
    if text == SOURCE_UPLOADED:
        return SOURCE_UPLOADED
    return SOURCE_UPLOADED


def source_type_label_fr(source_type: object | None) -> str:
    return "Externe" if normalize_source_type(source_type) == SOURCE_EXTERNAL else "Importé (ZIP)"


def external_root_exists(root: Path | None = None) -> bool:
    path = (root if root is not None else get_external_datasets_root())
    try:
        return path.is_dir()
    except OSError:
        return False


def is_under_root(path: Path, root: Path) -> bool:
    """Return True if ``path`` resolves strictly under ``root`` (symlink-safe)."""
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
    except OSError:
        return False
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        return False
    return True


def list_external_dataset_dirs(root: Path | None = None) -> list[Path]:
    """
    List first-level subdirectories under the external root.

    Does not recurse. Skips hidden names and non-directories.
    Returns an empty list when the root is missing.
    """
    base = root if root is not None else get_external_datasets_root()
    if not external_root_exists(base):
        return []
    entries: list[Path] = []
    try:
        children = sorted(base.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return []
    for child in children:
        name = child.name
        if name.startswith("."):
            continue
        try:
            if not child.is_dir():
                continue
            # Reject entries whose real path escapes the root (symlink jailbreak).
            if not is_under_root(child, base):
                continue
            entries.append(child)
        except OSError:
            continue
    return entries


def resolve_external_dataset_path(
    name: str,
    *,
    root: Path | None = None,
) -> Path:
    """
    Resolve a first-level dataset name under the external root.

    Rejects ``..``, absolute paths, and symlink escapes.
    """
    base = (root if root is not None else get_external_datasets_root())
    if not external_root_exists(base):
        raise ExternalDatasetError(
            f"La racine des datasets externes est inaccessible : {base}"
        )

    raw = (name or "").strip()
    if not raw:
        raise ExternalDatasetError("Nom de dataset externe vide.")
    if raw.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[/\\]", raw):
        raise ExternalDatasetError(
            "Les chemins absolus ne sont pas autorisés ; choisissez un dossier sous la racine."
        )
    if ".." in Path(raw).parts or "/" in raw or "\\" in raw:
        raise ExternalDatasetError(
            "Nom de dataset invalide (pas de sous-chemin ni de '..')."
        )
    if not _SAFE_NAME_RE.match(raw):
        raise ExternalDatasetError(
            f"Nom de dataset invalide : {raw!r}. "
            "Utilisez lettres, chiffres, '.', '_' ou '-'."
        )

    candidate = (base / raw)
    try:
        resolved = candidate.resolve()
        base_resolved = base.resolve()
    except OSError as exc:
        raise ExternalDatasetError(
            f"Impossible de résoudre le chemin du dataset : {exc}"
        ) from exc

    try:
        resolved.relative_to(base_resolved)
    except ValueError as exc:
        raise ExternalDatasetError(
            f"Le chemin sort de la racine autorisée ({base_resolved}) : {resolved}"
        ) from exc

    if not resolved.is_dir():
        raise ExternalDatasetError(
            f"Le dataset externe n'est plus accessible : {resolved}"
        )
    return resolved


def assert_external_dataset_accessible(path: Path) -> Path:
    """Ensure an already-selected external path still exists and stays under root."""
    root = get_external_datasets_root()
    try:
        resolved = path.resolve()
    except OSError as exc:
        raise ExternalDatasetError(
            f"Le dataset externe n'est plus accessible : {path}"
        ) from exc
    if not is_under_root(resolved, root):
        raise ExternalDatasetError(
            f"Le chemin sort de la racine autorisée ({root}) : {resolved}"
        )
    if not resolved.is_dir():
        raise ExternalDatasetError(
            f"Le dataset externe n'est plus accessible : {resolved}"
        )
    return resolved


def external_metadata_dir(display_name: str, *, task: str) -> Path:
    """Internal folder for small metadata (never holds images/labels copies)."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", display_name).strip("-") or "dataset"
    safe = safe[:80]
    from vision_trainer.tasks import normalize_task

    task_key = normalize_task(task)
    return get_datasets_dir() / f"ext-{task_key}-{safe}"


def register_external_dataset_meta(
    dataset_path: Path,
    *,
    task: str,
    display_name: str | None = None,
) -> tuple[str, Path]:
    """
    Create/update lightweight meta.json for an external dataset.

    Does **not** copy images or labels. Returns (dataset_id, meta_dir).
    """
    from vision_trainer.tasks import normalize_task

    resolved = assert_external_dataset_accessible(dataset_path)
    name = display_name or resolved.name
    task_key = normalize_task(task)
    meta_dir = external_metadata_dir(name, task=task_key)
    meta_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = f"ext-{task_key}-{name}"
    atomic_write_json(
        meta_dir / "meta.json",
        {
            "dataset_id": dataset_id,
            "source_type": SOURCE_EXTERNAL,
            "task": task_key,
            "name": name,
            "path": str(resolved),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return dataset_id, meta_dir


def generated_yaml_path_for_external(meta_dir: Path) -> Path:
    """Where auto-generated data.yaml is written (never inside the RO source)."""
    return meta_dir / "data.generated.yaml"
