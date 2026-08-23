from __future__ import annotations

from pathlib import Path

from vision_trainer.inference.models import AvailableModel
from vision_trainer.training.runs import ARTIFACTS_RUNS_DIR


def discover_trained_models(runs_root: Path | None = None) -> list[AvailableModel]:
    """
    Discover runs under ``artifacts/runs/`` that contain ``weights/best.pt``.

    Runs without ``best.pt`` are excluded. Results are sorted newest-first by folder name.
    """
    root = runs_root if runs_root is not None else ARTIFACTS_RUNS_DIR
    if not root.is_dir():
        return []

    models: list[AvailableModel] = []
    for child in sorted(root.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        best = child / "weights" / "best.pt"
        if not best.is_file():
            continue
        run_id = child.name
        models.append(
            AvailableModel(
                run_id=run_id,
                weights_path=str(best.resolve()),
                label=f"{run_id} — best.pt",
            )
        )
    return models
