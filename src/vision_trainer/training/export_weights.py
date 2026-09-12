"""User-facing named export of Ultralytics ``weights/best.pt`` (copy only)."""

from __future__ import annotations

import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any

from vision_trainer.tasks import TaskType, normalize_task

# Leave room for ``_{task}_best.pt`` (e.g. ``_classify_best.pt`` ≈ 18 chars).
MAX_DATASET_SLUG_LENGTH = 80

_EXPORT_SUFFIX_RE = re.compile(
    r"_(detect|classify|segment)_best\.pt$",
    re.IGNORECASE,
)


def sanitize_dataset_slug(
    name: str | None,
    *,
    max_length: int = MAX_DATASET_SLUG_LENGTH,
) -> str:
    """
    Turn a logical dataset name into a safe single-path-segment filename slug.

    - Uses the last path segment only (blocks traversal via ``/`` or ``\\``).
    - Strips accents, lowercases, maps non-alphanumerics to ``_``.
    - Never returns empty / ``.`` / ``..``.
    """
    raw = (name or "").strip()
    if not raw:
        return "dataset"

    # Path traversal / nested paths → basename only.
    candidate = raw.replace("\\", "/").rstrip("/")
    candidate = candidate.split("/")[-1].strip()
    if not candidate or candidate in {".", ".."}:
        return "dataset"

    normalized = unicodedata.normalize("NFKD", candidate)
    ascii_ish = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    lowered = ascii_ish.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug or slug in {".", ".."}:
        return "dataset"

    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("_") or "dataset"
    if "/" in slug or "\\" in slug or slug in {".", ".."}:
        return "dataset"
    return slug


def build_export_weights_filename(
    dataset_name: str | None,
    task: TaskType | str | None,
) -> str:
    """Return ``{slug}_{task}_best.pt`` (never ``best.pt`` / ``last.pt``)."""
    task_key = normalize_task(task)
    slug = sanitize_dataset_slug(dataset_name)
    return f"{slug}_{task_key}_best.pt"


def resolve_export_dataset_name(request: dict[str, Any] | None) -> str:
    """Logical dataset name from ``request.json`` (ZIP, external, or legacy)."""
    if not isinstance(request, dict):
        return "dataset"

    name = request.get("dataset_name")
    if name is not None and str(name).strip():
        return str(name).strip()

    for key in ("dataset_root", "data_dir"):
        root = request.get(key)
        if root:
            basename = Path(str(root)).name.strip()
            if basename and basename not in {".", ".."}:
                return basename
    return "dataset"


def export_named_best_weights(
    run_dir: Path,
    *,
    dataset_name: str | None,
    task: TaskType | str | None,
    best_path: Path | None = None,
) -> Path | None:
    """
    Copy ``weights/best.pt`` to a user-friendly name beside it.

    Does not rename or delete the Ultralytics canonical ``best.pt``.
    Returns the export path, or ``None`` if ``best.pt`` is missing.
    """
    best = best_path if best_path is not None else (run_dir / "weights" / "best.pt")
    best = Path(best)
    if not best.is_file():
        return None

    weights_dir = (run_dir / "weights").resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)

    filename = build_export_weights_filename(dataset_name, task)
    if filename in {"best.pt", "last.pt"}:
        filename = build_export_weights_filename("dataset", task)

    dest = (weights_dir / filename).resolve()
    try:
        dest.relative_to(weights_dir)
    except ValueError:
        # Should be unreachable after sanitization; refuse to write outside.
        safe_name = build_export_weights_filename("dataset", task)
        dest = (weights_dir / safe_name).resolve()

    shutil.copy2(best, dest)
    return dest


def find_export_weights(
    run_dir: Path,
    *,
    export_model_path: str | None = None,
) -> Path | None:
    """Locate a named export next to ``best.pt`` (status path or filename scan)."""
    if export_model_path:
        candidate = Path(export_model_path)
        if candidate.is_file():
            return candidate

    weights_dir = run_dir / "weights"
    if not weights_dir.is_dir():
        return None

    matches = sorted(
        path
        for path in weights_dir.iterdir()
        if path.is_file()
        and path.name not in {"best.pt", "last.pt"}
        and _EXPORT_SUFFIX_RE.search(path.name)
    )
    return matches[0] if matches else None
