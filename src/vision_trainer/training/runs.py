from __future__ import annotations

import secrets
from datetime import datetime, timezone
from pathlib import Path

from vision_trainer.paths import get_runs_dir


def __getattr__(name: str) -> Path:
    """Lazy ``ARTIFACTS_RUNS_DIR`` so ``VISIONTRAINER_DATA_DIR`` is respected."""
    if name == "ARTIFACTS_RUNS_DIR":
        return get_runs_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def generate_run_id(now: datetime | None = None) -> str:
    """Create a unique run id: YYYYMMDD-HHMMSS-<short>."""
    moment = now or datetime.now(timezone.utc).astimezone()
    stamp = moment.strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(3)
    return f"{stamp}-{suffix}"


def create_run_directory(
    run_id: str | None = None,
    *,
    runs_root: Path | None = None,
) -> tuple[str, Path]:
    """
    Create a dedicated artifacts directory for one training run.

    Never reuses an existing directory: if the path already exists, a new id is generated.
    """
    root = runs_root if runs_root is not None else get_runs_dir()
    root.mkdir(parents=True, exist_ok=True)

    chosen = run_id or generate_run_id()
    run_dir = root / chosen
    while run_dir.exists():
        chosen = generate_run_id()
        run_dir = root / chosen

    run_dir.mkdir(parents=False, exist_ok=False)
    return chosen, run_dir
