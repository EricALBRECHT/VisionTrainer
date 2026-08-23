"""Subprocess entrypoint for Ultralytics training runs."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from vision_trainer.training.status import read_status, utc_now_iso, write_status
from vision_trainer.training.trainer import TrainingError, execute_training_from_run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vision Trainer YOLO training worker")
    parser.add_argument("--run-dir", required=True, type=Path, help="Prepared run directory")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"Run directory introuvable : {run_dir}", file=sys.stderr)
        return 2

    try:
        execute_training_from_run_dir(run_dir)
    except TrainingError as exc:
        status = read_status(run_dir)
        if status is not None and status.state not in {"failed", "completed"}:
            status.state = "failed"
            status.finished_at = utc_now_iso()
            status.error_message = str(exc)
            write_status(run_dir, status)
        print(f"TrainingError: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        status = read_status(run_dir)
        if status is not None:
            status.state = "failed"
            status.finished_at = utc_now_iso()
            status.error_message = f"Erreur worker : {exc}"
            write_status(run_dir, status)
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
