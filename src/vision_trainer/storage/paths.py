from __future__ import annotations

from pathlib import Path


def path_is_within(path: Path, root: Path) -> bool:
    """Return True when ``path`` resolves strictly inside ``root`` (or equals it)."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def assert_path_allowed(path: Path, allowed_roots: list[Path]) -> Path:
    """
    Resolve ``path`` and ensure it lies under one of ``allowed_roots``.

    Raises ``PermissionError`` when the path escapes every allowed root.
    """
    resolved = path.resolve()
    for root in allowed_roots:
        if path_is_within(resolved, root):
            return resolved
    raise PermissionError(
        f"Chemin hors des répertoires autorisés : {resolved}"
    )


def default_allowed_roots(
    *,
    artifacts_root: Path | None = None,
    datasets_root: Path | None = None,
    runs_root: Path | None = None,
) -> list[Path]:
    """Authorized deletion / scan roots for Vision Trainer."""
    from vision_trainer.datasets.store import ARTIFACTS_DATASETS_DIR
    from vision_trainer.training.runs import ARTIFACTS_RUNS_DIR

    if artifacts_root is not None:
        root = artifacts_root.resolve()
        return [root]

    roots = [
        (datasets_root if datasets_root is not None else ARTIFACTS_DATASETS_DIR).resolve(),
        (runs_root if runs_root is not None else ARTIFACTS_RUNS_DIR).resolve(),
    ]
    # Deduplicate while preserving order.
    unique: list[Path] = []
    seen: set[Path] = set()
    for item in roots:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique
