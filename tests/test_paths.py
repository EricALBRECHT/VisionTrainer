from __future__ import annotations

from pathlib import Path

from vision_trainer.paths import (
    DATA_DIR_ENV,
    get_data_root,
    get_datasets_dir,
    get_runs_dir,
    get_tmp_dir,
)
from vision_trainer.training import runs as runs_mod
from vision_trainer.datasets import store as store_mod


def test_default_data_root_is_artifacts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert get_data_root() == (tmp_path / "artifacts").resolve()
    assert get_datasets_dir() == (tmp_path / "artifacts" / "datasets").resolve()
    assert get_runs_dir() == (tmp_path / "artifacts" / "runs").resolve()


def test_env_overrides_data_root(monkeypatch, tmp_path: Path) -> None:
    data = tmp_path / "custom-data"
    monkeypatch.setenv(DATA_DIR_ENV, str(data))
    assert get_data_root() == data.resolve()
    assert get_datasets_dir() == (data / "datasets").resolve()
    assert get_runs_dir() == (data / "runs").resolve()
    assert runs_mod.ARTIFACTS_RUNS_DIR == (data / "runs").resolve()
    assert store_mod.ARTIFACTS_DATASETS_DIR == (data / "datasets").resolve()


def test_get_tmp_dir_creates(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "d"))
    tmp = get_tmp_dir()
    assert tmp.is_dir()
    assert tmp == (tmp_path / "d" / "tmp").resolve()
