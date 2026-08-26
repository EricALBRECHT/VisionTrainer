from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone
from pathlib import Path

from vision_trainer.io_utils import atomic_write_json
from vision_trainer.paths import get_datasets_dir
from vision_trainer.yolo.parser import extract_zip_dataset


def __getattr__(name: str) -> Path:
    """Lazy ``ARTIFACTS_DATASETS_DIR`` so ``VISIONTRAINER_DATA_DIR`` is respected."""
    if name == "ARTIFACTS_DATASETS_DIR":
        return get_datasets_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def generate_dataset_id(content_hash: str | None = None) -> str:
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    suffix = (content_hash or secrets.token_hex(4))[:8]
    return f"{stamp}-{suffix}"


def import_zip_to_persistent_dataset(
    zip_bytes: bytes,
    *,
    datasets_root: Path | None = None,
    content_hash: str | None = None,
) -> tuple[str, Path]:
    """
    Persist an uploaded YOLO ZIP under ``<data_root>/datasets/<dataset_id>/extracted``.

    Returns (dataset_id, extract_dir). Does not delete previous datasets.
    """
    root = datasets_root if datasets_root is not None else get_datasets_dir()
    digest = content_hash or hashlib.sha256(zip_bytes).hexdigest()
    dataset_id = generate_dataset_id(digest)
    dataset_dir = root / dataset_id
    extract_dir = dataset_dir / "extracted"
    dataset_dir.mkdir(parents=True, exist_ok=False)
    extract_dir.mkdir(parents=True, exist_ok=True)

    zip_path = dataset_dir / "source.zip"
    zip_path.write_bytes(zip_bytes)

    try:
        extract_zip_dataset(zip_path, extract_dir)
    except Exception:
        # Keep the failed folder for inspection; re-raise for the UI.
        raise

    atomic_write_json(
        dataset_dir / "meta.json",
        {
            "dataset_id": dataset_id,
            "content_sha256": digest,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "extract_dir": str(extract_dir.resolve()),
        },
    )
    return dataset_id, extract_dir.resolve()


def dataset_is_referenced_by_active_run(
    dataset_id: str,
    runs_root: Path,
) -> bool:
    """Return True when an active/reserved run still points at this dataset_id."""
    from vision_trainer.training.status import find_reserved_run, read_request_safe, read_status

    reserved = find_reserved_run(runs_root)
    if reserved is None:
        return False
    status = read_status(reserved)
    if status is None:
        return False
    request = read_request_safe(reserved)
    if request and request.get("dataset_id") == dataset_id:
        return True
    return False
