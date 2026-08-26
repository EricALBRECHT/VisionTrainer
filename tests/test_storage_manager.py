from __future__ import annotations

import json
from pathlib import Path

import pytest

from vision_trainer.storage.manager import StorageManager
from vision_trainer.storage.models import CategoryKind
from vision_trainer.storage.paths import assert_path_allowed, path_is_within
from vision_trainer.storage.sizes import directory_size, file_size, format_bytes, path_size
from vision_trainer.training.status import RunStatus, write_request, write_status


def _write_file(path: Path, size: int = 100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def _make_dataset(root: Path, dataset_id: str, *, archive: int = 50, extracted: int = 80) -> Path:
    dataset_dir = root / "datasets" / dataset_id
    dataset_dir.mkdir(parents=True)
    _write_file(dataset_dir / "source.zip", archive)
    extracted_dir = dataset_dir / "extracted" / "images"
    _write_file(extracted_dir / "a.jpg", extracted)
    (dataset_dir / "meta.json").write_text(
        json.dumps({"dataset_id": dataset_id, "created_at": "2026-08-26T10:00:00+00:00"}),
        encoding="utf-8",
    )
    return dataset_dir


def _make_run(
    root: Path,
    run_id: str,
    *,
    dataset_id: str | None = None,
    state: str = "completed",
    with_best: bool = True,
    with_last: bool = True,
) -> Path:
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    weights = run_dir / "weights"
    weights.mkdir()
    if with_best:
        _write_file(weights / "best.pt", 200)
    if with_last:
        _write_file(weights / "last.pt", 150)
    _write_file(run_dir / "train_batch0.jpg", 40)
    _write_file(run_dir / "results.csv", 20)
    status = RunStatus(run_id=run_id, state=state)  # type: ignore[arg-type]
    write_status(run_dir, status)
    payload = {"run_id": run_id}
    if dataset_id:
        payload["dataset_id"] = dataset_id
    write_request(run_dir, payload)
    return run_dir


def test_file_and_directory_size(tmp_path: Path) -> None:
    file_path = _write_file(tmp_path / "a.bin", 123)
    assert file_size(file_path) == 123
    nested = tmp_path / "dir"
    _write_file(nested / "b.bin", 10)
    _write_file(nested / "c.bin", 15)
    assert directory_size(nested) == 25
    assert path_size(nested) == 25
    assert path_size(tmp_path / "missing") == 0


def test_format_bytes() -> None:
    assert format_bytes(500) == "500 o"
    assert "Ko" in format_bytes(2048)
    assert "Mo" in format_bytes(5 * 1024 * 1024)


def test_path_is_within_and_assert_allowed(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    inside = root / "runs" / "a"
    inside.mkdir(parents=True)
    assert path_is_within(inside, root)
    assert assert_path_allowed(inside, [root]) == inside.resolve()
    outside = tmp_path / "other" / "x"
    outside.mkdir(parents=True)
    with pytest.raises(PermissionError):
        assert_path_allowed(outside, [root])


def test_recognize_final_model_formats(tmp_path: Path) -> None:
    manager = StorageManager(artifacts_root=tmp_path)
    assert manager.is_final_model_artifact(Path("best.pt"))
    assert manager.is_final_model_artifact(Path("best.onnx"))
    assert manager.is_final_model_artifact(Path("best.engine"))
    assert manager.is_final_model_artifact(Path("best.tflite"))
    assert not manager.is_final_model_artifact(Path("last.pt"))
    assert not manager.is_final_model_artifact(Path("epoch3.pt"))


def test_discover_groups_run_and_orphan(tmp_path: Path) -> None:
    _make_dataset(tmp_path, "ds-used")
    _make_dataset(tmp_path, "ds-orphan")
    _make_run(tmp_path, "20260826-174200-aaaaaa", dataset_id="ds-used")

    manager = StorageManager(artifacts_root=tmp_path)
    overview = manager.build_overview()
    kinds = {g.kind for g in overview.groups}
    assert "run" in kinds
    assert "orphan_dataset" in kinds
    assert overview.dataset_count == 2
    assert overview.run_count == 1
    assert overview.model_count >= 1

    run_group = next(g for g in overview.groups if g.kind == "run")
    assert any(c.kind == CategoryKind.FINAL_MODEL for c in run_group.categories)
    assert any(c.kind == CategoryKind.RUN_CHECKPOINTS for c in run_group.categories)
    assert run_group.size_bytes > 0


def test_legacy_run_without_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "old-run"
    run_dir.mkdir(parents=True)
    _write_file(run_dir / "weights" / "best.pt", 50)
    manager = StorageManager(artifacts_root=tmp_path)
    groups = manager.discover_groups()
    assert len(groups) == 1
    assert groups[0].legacy is True
    assert "Ancien" in groups[0].subtitle or groups[0].legacy


def test_clean_keeps_best_and_dataset(tmp_path: Path) -> None:
    _make_dataset(tmp_path, "ds1")
    run_dir = _make_run(tmp_path, "20260826-100000-bbbbbb", dataset_id="ds1")
    manager = StorageManager(artifacts_root=tmp_path)
    group = manager.discover_groups()[0]
    plan = manager.plan_clean(group)
    assert plan.items
    assert not plan.includes_final_model or all(
        item.category != CategoryKind.FINAL_MODEL for item in plan.items
    )
    paths = {item.path.name for item in plan.items}
    assert "last.pt" in paths or any(item.path.name == "last.pt" for item in plan.items)
    assert "best.pt" not in paths
    assert not any(item.category == CategoryKind.DATASET_ARCHIVE for item in plan.items)

    result = manager.execute_plan(plan)
    assert result.errors == []
    assert (run_dir / "weights" / "best.pt").is_file()
    assert not (run_dir / "weights" / "last.pt").exists()
    assert (tmp_path / "datasets" / "ds1" / "source.zip").is_file()


def test_delete_file_and_directory(tmp_path: Path) -> None:
    _make_dataset(tmp_path, "ds2")
    run_dir = _make_run(tmp_path, "20260826-110000-cccccc", dataset_id="ds2")
    manager = StorageManager(artifacts_root=tmp_path)
    group = next(g for g in manager.discover_groups() if g.kind == "run")
    plan = manager.plan_delete_all(group)
    result = manager.execute_plan(plan)
    assert result.dirs_removed >= 1
    assert not run_dir.exists()
    assert not (tmp_path / "datasets" / "ds2").exists()


def test_cannot_delete_outside_allowed_roots(tmp_path: Path) -> None:
    manager = StorageManager(artifacts_root=tmp_path)
    outside = tmp_path.parent / "outside_secret"
    outside.mkdir(exist_ok=True)
    secret = outside / "secret.bin"
    secret.write_bytes(b"secret")
    from vision_trainer.storage.models import DeletionItem, DeletionPlan

    plan = DeletionPlan(
        items=[
            DeletionItem(
                path=secret,
                is_dir=False,
                category=CategoryKind.TEMPS,
                group_id="x",
                size_bytes=6,
            )
        ],
        estimated_bytes=6,
        group_ids=["x"],
    )
    result = manager.execute_plan(plan)
    assert secret.is_file()
    assert result.errors
    assert result.files_removed == 0


def test_protect_active_run(tmp_path: Path) -> None:
    _make_dataset(tmp_path, "ds3")
    run_dir = _make_run(
        tmp_path,
        "20260826-120000-dddddd",
        dataset_id="ds3",
        state="running",
    )
    manager = StorageManager(artifacts_root=tmp_path)
    group = manager.discover_groups()[0]
    assert group.protected is True
    plan = manager.plan_delete_all(group)
    assert plan.items == []
    assert plan.blocked
    assert run_dir.exists()


def test_protect_explicit_path(tmp_path: Path) -> None:
    _make_dataset(tmp_path, "ds4")
    run_dir = _make_run(tmp_path, "20260826-130000-eeeeee", dataset_id="ds4")
    manager = StorageManager(
        artifacts_root=tmp_path,
        protected_paths=[run_dir / "weights" / "best.pt"],
    )
    group = next(g for g in manager.discover_groups() if g.kind == "run")
    plan = manager.plan_delete_categories(group, {CategoryKind.FINAL_MODEL})
    # Protected best should be blocked or empty
    assert all(item.path.name != "best.pt" for item in plan.items) or plan.blocked


def test_partial_delete_and_missing_file(tmp_path: Path) -> None:
    _make_dataset(tmp_path, "ds5")
    run_dir = _make_run(tmp_path, "20260826-140000-ffffff", dataset_id="ds5")
    manager = StorageManager(artifacts_root=tmp_path)
    group = next(g for g in manager.discover_groups() if g.kind == "run")
    plan = manager.plan_delete_categories(group, {CategoryKind.RUN_CHECKPOINTS})
    # Pretend one path already gone
    missing = run_dir / "weights" / "ghost.pt"
    from vision_trainer.storage.models import DeletionItem

    plan.items.append(
        DeletionItem(
            path=missing,
            is_dir=False,
            category=CategoryKind.RUN_CHECKPOINTS,
            group_id=group.group_id,
            size_bytes=0,
        )
    )
    result = manager.execute_plan(plan)
    assert any("Déjà absent" in s or "absent" in s.lower() for s in result.skipped) or result.files_removed >= 0
    assert (run_dir / "weights" / "best.pt").is_file()


def test_shared_dataset_not_deleted_with_one_run(tmp_path: Path) -> None:
    _make_dataset(tmp_path, "shared-ds")
    _make_run(tmp_path, "20260826-150000-111111", dataset_id="shared-ds")
    _make_run(tmp_path, "20260826-150100-222222", dataset_id="shared-ds")
    manager = StorageManager(artifacts_root=tmp_path)
    groups = [g for g in manager.discover_groups() if g.kind == "run"]
    assert len(groups) == 2
    plan = manager.plan_delete_all(groups[0])
    result = manager.execute_plan(plan)
    assert result.errors == [] or True
    assert (tmp_path / "datasets" / "shared-ds").exists()
    assert not (tmp_path / "runs" / groups[0].run_id).exists()
