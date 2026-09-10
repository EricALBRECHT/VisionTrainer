from __future__ import annotations

from pathlib import Path

from vision_trainer.inference.models import AvailableModel
from vision_trainer.tasks import TaskType, normalize_task
from vision_trainer.training.runs import ARTIFACTS_RUNS_DIR
from vision_trainer.training.status import read_request_safe, read_status


def discover_trained_models(
    runs_root: Path | None = None,
    *,
    task: TaskType | str | None = None,
) -> list[AvailableModel]:
    """
    Discover runs under ``artifacts/runs/`` that contain ``weights/best.pt``.

    When ``task`` is set, only runs matching that task are returned. Legacy runs
    without a ``task`` field are treated as ``detect``.
    """
    root = runs_root if runs_root is not None else ARTIFACTS_RUNS_DIR
    if not root.is_dir():
        return []

    wanted = normalize_task(task) if task is not None else None
    models: list[AvailableModel] = []
    for child in sorted(root.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        best = child / "weights" / "best.pt"
        if not best.is_file():
            continue
        run_task = _run_task(child)
        if wanted is not None and run_task != wanted:
            continue
        run_id = child.name
        task_tag = (
            "cls"
            if run_task == "classify"
            else "seg"
            if run_task == "segment"
            else "det"
            if run_task == "detect"
            else run_task
        )
        models.append(
            AvailableModel(
                run_id=run_id,
                weights_path=str(best.resolve()),
                label=f"{run_id} [{task_tag}] — best.pt",
            )
        )
    return models


def _run_task(run_dir: Path) -> TaskType:
    status = read_status(run_dir)
    if status is not None:
        return normalize_task(status.task)
    request = read_request_safe(run_dir)
    if request is not None:
        return normalize_task(request.get("task"))
    return normalize_task(None)
