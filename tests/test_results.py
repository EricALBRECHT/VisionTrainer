from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from vision_trainer.results.catalog import (
    discover_runs,
    format_optional,
    load_metrics_history,
    load_run,
)


def _make_run(
    root: Path,
    run_id: str,
    *,
    status: dict | None = None,
    with_best: bool = True,
    with_last: bool = True,
    with_csv: bool = False,
    csv_header: str | None = None,
    csv_rows: list[str] | None = None,
    plots: list[str] | None = None,
    data_yaml: str | None = None,
    invalid_status: bool = False,
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    weights = run_dir / "weights"
    weights.mkdir()
    if with_best:
        (weights / "best.pt").write_bytes(b"best")
    if with_last:
        (weights / "last.pt").write_bytes(b"last")

    if invalid_status:
        (run_dir / "status.json").write_text("{not-json", encoding="utf-8")
    elif status is not None:
        (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")

    if with_csv:
        header = csv_header or "epoch,metrics/mAP50(B),metrics/mAP50-95(B)"
        rows = csv_rows or ["1,0.5,0.4", "2,0.6,0.45", "3,0.7,0.5"]
        (run_dir / "results.csv").write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

    for plot_name in plots or []:
        Image.new("RGB", (8, 8), (100, 100, 100)).save(run_dir / plot_name)

    if data_yaml is not None:
        (run_dir / "data.resolved.yaml").write_text(data_yaml, encoding="utf-8")

    return run_dir


def test_discover_multiple_runs_newest_first(tmp_path: Path) -> None:
    _make_run(
        tmp_path,
        "20260822-100000-aaaaaa",
        status={
            "run_id": "20260822-100000-aaaaaa",
            "state": "completed",
            "model": "YOLO11n",
            "epochs_total": 3,
            "started_at": "2026-08-22T10:00:00+00:00",
            "finished_at": "2026-08-22T10:10:00+00:00",
        },
    )
    _make_run(
        tmp_path,
        "20260822-180000-bbbbbb",
        status={
            "run_id": "20260822-180000-bbbbbb",
            "state": "completed",
            "model": "YOLO11s",
            "epochs_total": 5,
            "started_at": "2026-08-22T18:00:00+00:00",
            "finished_at": "2026-08-22T18:30:00+00:00",
        },
    )

    runs = discover_runs(tmp_path)
    assert [item.run_id for item in runs] == [
        "20260822-180000-bbbbbb",
        "20260822-100000-aaaaaa",
    ]
    assert runs[0].state == "terminé"
    assert runs[0].model == "YOLO11s"
    assert runs[0].duration_seconds == 1800


def test_load_complete_run(tmp_path: Path) -> None:
    run_dir = _make_run(
        tmp_path,
        "run-complete",
        status={
            "run_id": "run-complete",
            "state": "completed",
            "model": "YOLO11n",
            "epochs_total": 3,
            "imgsz": 640,
            "batch": -1,
            "device": "cpu",
            "started_at": "2026-08-22T10:00:00+00:00",
            "finished_at": "2026-08-22T10:05:00+00:00",
            "metrics": {
                "precision": 0.9,
                "recall": 0.8,
                "map50": 0.75,
                "map50_95": 0.55,
            },
        },
        plots=["results.png", "PR_curve.png"],
        data_yaml="nc: 1\nnames: [pothole]\n",
        with_csv=True,
    )

    detail = load_run(run_dir)
    assert detail.summary.state == "terminé"
    assert detail.summary.num_classes == 1
    assert detail.precision == 0.9
    assert detail.map50 == 0.75
    assert detail.best_pt is not None
    assert detail.last_pt is not None
    assert "results.png" in detail.plots
    assert "PR_curve.png" in detail.plots
    assert "F1_curve.png" not in detail.plots


def test_incomplete_run_and_missing_status(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, "run-incomplete", status=None, with_best=False, with_last=False)
    summary = discover_runs(tmp_path)[0]
    assert summary.run_id == "run-incomplete"
    assert summary.has_best is False
    assert summary.has_status is False
    assert summary.state in {"incomplet", "inconnu"}

    detail = load_run(run_dir)
    assert detail.best_pt is None
    assert detail.map50 is None
    assert format_optional(detail.map50) == "Non disponible"


def test_invalid_status_json(tmp_path: Path) -> None:
    _make_run(tmp_path, "run-bad-status", invalid_status=True, with_best=True)
    runs = discover_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].has_status is True
    assert runs[0].load_warning is not None
    assert runs[0].has_best is True


def test_best_pt_present_absent(tmp_path: Path) -> None:
    _make_run(tmp_path, "with-best", status={"run_id": "with-best", "state": "completed"}, with_best=True)
    _make_run(tmp_path, "no-best", status={"run_id": "no-best", "state": "failed"}, with_best=False)

    by_id = {item.run_id: item for item in discover_runs(tmp_path)}
    assert by_id["with-best"].has_best is True
    assert by_id["no-best"].has_best is False
    assert load_run(tmp_path / "no-best").best_pt is None


def test_results_csv_map_extraction(tmp_path: Path) -> None:
    run_dir = _make_run(
        tmp_path,
        "csv-run",
        status={"run_id": "csv-run", "state": "completed"},
        with_csv=True,
        csv_header="epoch, metrics/mAP50(B) , metrics/mAP50-95(B) ",
        csv_rows=["1,0.41,0.31", "2,0.52,0.39"],
    )
    history = load_metrics_history(run_dir)
    assert history is not None
    assert history.epochs == [1, 2]
    assert history.map50 == [0.41, 0.52]
    assert history.map50_95 == [0.31, 0.39]
    assert history.has_map50 is True
    assert history.has_map50_95 is True


def test_results_csv_absent(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, "no-csv", status={"run_id": "no-csv", "state": "completed"})
    assert load_metrics_history(run_dir) is None


def test_plot_present_absent(tmp_path: Path) -> None:
    run_dir = _make_run(
        tmp_path,
        "plots-run",
        status={"run_id": "plots-run", "state": "completed"},
        plots=["confusion_matrix.png"],
    )
    detail = load_run(run_dir)
    assert "confusion_matrix.png" in detail.plots
    assert "results.png" not in detail.plots


def test_corrupted_run_does_not_block_others(tmp_path: Path) -> None:
    _make_run(
        tmp_path,
        "20260823-120000-good",
        status={"run_id": "20260823-120000-good", "state": "completed", "model": "YOLO11n"},
    )
    bad = tmp_path / "20260823-110000-bad"
    bad.mkdir()
    # Make directory unreadable via a weird file that still allows listing;
    # simulate load failure by replacing with a file named like a dir? Instead create
    # a directory that raises during summary via invalid symlink-like content.
    # We'll create a run whose status triggers warning but still returns.
    (bad / "status.json").write_text("{broken", encoding="utf-8")

    runs = discover_runs(tmp_path)
    ids = {item.run_id for item in runs}
    assert "20260823-120000-good" in ids
    assert "20260823-110000-bad" in ids
    good = next(item for item in runs if item.run_id == "20260823-120000-good")
    assert good.model == "YOLO11n"
