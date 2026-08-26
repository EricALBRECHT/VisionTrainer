from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

from vision_trainer.storage.models import (
    CATEGORY_LABELS,
    CLEANABLE_CATEGORIES,
    CategoryKind,
    DeletionItem,
    DeletionPlan,
    DeletionResult,
    StorageCategory,
    StorageGroup,
    StorageOverview,
    StoragePathRef,
)
from vision_trainer.storage.paths import assert_path_allowed, default_allowed_roots, path_is_within
from vision_trainer.storage.sizes import format_bytes, path_size
from vision_trainer.training.status import (
    ACTIVE_STATES,
    read_request_safe,
    read_status,
)

# Final model artifacts (not limited to YOLO .pt).
FINAL_MODEL_STEMS = frozenset({"best"})
FINAL_MODEL_SUFFIXES = frozenset(
    {".pt", ".pth", ".onnx", ".engine", ".tflite", ".pb", ".mlmodel", ".bin", ".xml"}
)

CHECKPOINT_NAME_PREFIXES = ("epoch", "last")
TEMP_NAME_MARKERS = ("__pycache__", ".cache", ".tmp", ".raw.mp4")
TEMP_FILE_GLOBS = ("train_batch*.jpg", "labels.jpg")


class StorageManager:
    """Discover, classify and safely delete Vision Trainer artifacts."""

    def __init__(
        self,
        *,
        datasets_root: Path | None = None,
        runs_root: Path | None = None,
        artifacts_root: Path | None = None,
        protected_paths: Iterable[Path] | None = None,
    ) -> None:
        from vision_trainer.datasets.store import ARTIFACTS_DATASETS_DIR
        from vision_trainer.training.runs import ARTIFACTS_RUNS_DIR

        if artifacts_root is not None:
            self.artifacts_root = artifacts_root.resolve()
            self.datasets_root = (artifacts_root / "datasets").resolve()
            self.runs_root = (artifacts_root / "runs").resolve()
        else:
            self.artifacts_root = None
            self.datasets_root = (
                datasets_root if datasets_root is not None else ARTIFACTS_DATASETS_DIR
            ).resolve()
            self.runs_root = (
                runs_root if runs_root is not None else ARTIFACTS_RUNS_DIR
            ).resolve()

        self.allowed_roots = default_allowed_roots(
            artifacts_root=self.artifacts_root,
            datasets_root=self.datasets_root,
            runs_root=self.runs_root,
        )
        self.protected_paths = {Path(p).resolve() for p in (protected_paths or [])}

    # ------------------------------------------------------------------ overview
    def build_overview(self) -> StorageOverview:
        groups = self.discover_groups()
        model_count = sum(
            1
            for group in groups
            for category in group.categories
            if category.kind == CategoryKind.FINAL_MODEL and category.size_bytes > 0
        )
        dataset_ids = {
            group.dataset_id
            for group in groups
            if group.dataset_id and (group.dataset_dir is None or group.dataset_dir.exists())
        }
        # Count physical dataset directories.
        dataset_count = 0
        if self.datasets_root.is_dir():
            dataset_count = sum(1 for p in self.datasets_root.iterdir() if p.is_dir())

        run_count = sum(1 for group in groups if group.kind == "run")
        total = sum(group.size_bytes for group in groups)
        # Avoid double-counting shared datasets across run groups.
        total = self._estimate_unique_bytes(groups)

        free = None
        try:
            usage = shutil.disk_usage(str(self.runs_root if self.runs_root.exists() else Path.cwd()))
            free = int(usage.free)
        except OSError:
            free = None

        return StorageOverview(
            total_bytes=total,
            dataset_count=dataset_count,
            run_count=run_count,
            model_count=model_count,
            free_disk_bytes=free,
            groups=groups,
        )

    def discover_groups(self) -> list[StorageGroup]:
        groups: list[StorageGroup] = []
        referenced_datasets: set[str] = set()

        if self.runs_root.is_dir():
            for run_dir in sorted(self.runs_root.iterdir(), reverse=True):
                if not run_dir.is_dir():
                    continue
                if run_dir.name.startswith("."):
                    continue
                group = self._build_run_group(run_dir)
                if group.dataset_id:
                    referenced_datasets.add(group.dataset_id)
                groups.append(group)

        if self.datasets_root.is_dir():
            for dataset_dir in sorted(self.datasets_root.iterdir(), reverse=True):
                if not dataset_dir.is_dir():
                    continue
                if dataset_dir.name in referenced_datasets:
                    continue
                groups.append(self._build_orphan_dataset_group(dataset_dir))

        return groups

    # ------------------------------------------------------------------ classify
    def is_final_model_artifact(self, path: Path) -> bool:
        """Recognize final model files across common export formats."""
        name = path.name.lower()
        stem = path.stem.lower()
        suffix = path.suffix.lower()
        if stem in FINAL_MODEL_STEMS and (suffix in FINAL_MODEL_SUFFIXES or suffix == ""):
            return True
        # best.pt, best.onnx, best_model.pt style
        if name.startswith("best.") and suffix in FINAL_MODEL_SUFFIXES:
            return True
        if stem.startswith("best_") and suffix in FINAL_MODEL_SUFFIXES:
            return True
        return False

    # ------------------------------------------------------------------ plans
    def plan_delete_categories(
        self,
        group: StorageGroup,
        category_kinds: Iterable[CategoryKind],
    ) -> DeletionPlan:
        kinds = set(category_kinds)
        items: list[DeletionItem] = []
        blocked: list[str] = []

        if group.protected:
            return DeletionPlan(
                blocked=[group.protection_reason or "Groupe protégé (entraînement actif)."]
            )

        for category in group.categories:
            if category.kind not in kinds:
                continue
            if category.kind in {
                CategoryKind.DATASET_ARCHIVE,
                CategoryKind.DATASET_EXTRACTED,
            } and category.shared:
                blocked.append(
                    f"{category.label} : dataset partagé avec d'autres runs — "
                    "suppression refusée depuis ce groupe."
                )
                continue
            for ref in category.paths:
                try:
                    resolved = assert_path_allowed(ref.path, self.allowed_roots)
                except PermissionError as exc:
                    blocked.append(str(exc))
                    continue
                if self._is_protected(resolved):
                    blocked.append(f"Protégé (actif) : {resolved}")
                    continue
                if not resolved.exists():
                    continue
                items.append(
                    DeletionItem(
                        path=resolved,
                        is_dir=resolved.is_dir(),
                        category=category.kind,
                        group_id=group.group_id,
                        size_bytes=path_size(resolved),
                    )
                )

        return self._finalize_plan(items, blocked, [group.group_id])

    def plan_clean(self, group: StorageGroup) -> DeletionPlan:
        return self.plan_delete_categories(group, CLEANABLE_CATEGORIES)

    def plan_delete_dataset(self, group: StorageGroup) -> DeletionPlan:
        return self.plan_delete_categories(
            group,
            {CategoryKind.DATASET_ARCHIVE, CategoryKind.DATASET_EXTRACTED},
        )

    def plan_delete_all(self, group: StorageGroup) -> DeletionPlan:
        # Prefer deleting entire run_dir / dataset_dir when exclusive.
        if group.protected:
            return DeletionPlan(
                blocked=[group.protection_reason or "Groupe protégé (entraînement actif)."]
            )

        items: list[DeletionItem] = []
        blocked: list[str] = []

        if group.run_dir is not None and group.run_dir.exists():
            try:
                resolved = assert_path_allowed(group.run_dir, self.allowed_roots)
            except PermissionError as exc:
                blocked.append(str(exc))
            else:
                if self._is_protected(resolved):
                    blocked.append(f"Protégé (actif) : {resolved}")
                else:
                    items.append(
                        DeletionItem(
                            path=resolved,
                            is_dir=True,
                            category=CategoryKind.RUN_ARTIFACTS,
                            group_id=group.group_id,
                            size_bytes=path_size(resolved),
                        )
                    )

        if group.dataset_dir is not None and group.dataset_dir.exists():
            shared = self._dataset_reference_count(group.dataset_id or "") > (
                1 if group.kind == "run" else 0
            )
            if shared:
                blocked.append(
                    "Dataset partagé avec d'autres runs — non inclus dans « Supprimer tout »."
                )
            else:
                try:
                    resolved = assert_path_allowed(group.dataset_dir, self.allowed_roots)
                except PermissionError as exc:
                    blocked.append(str(exc))
                else:
                    if self._is_protected(resolved):
                        blocked.append(f"Protégé (actif) : {resolved}")
                    else:
                        items.append(
                            DeletionItem(
                                path=resolved,
                                is_dir=True,
                                category=CategoryKind.DATASET_EXTRACTED,
                                group_id=group.group_id,
                                size_bytes=path_size(resolved),
                            )
                        )

        # Orphan dataset-only groups without run_dir.
        if group.kind == "orphan_dataset" and not items and group.dataset_dir:
            try:
                resolved = assert_path_allowed(group.dataset_dir, self.allowed_roots)
                items.append(
                    DeletionItem(
                        path=resolved,
                        is_dir=True,
                        category=CategoryKind.DATASET_EXTRACTED,
                        group_id=group.group_id,
                        size_bytes=path_size(resolved),
                    )
                )
            except PermissionError as exc:
                blocked.append(str(exc))

        return self._finalize_plan(items, blocked, [group.group_id])

    def merge_plans(self, plans: list[DeletionPlan]) -> DeletionPlan:
        items: list[DeletionItem] = []
        blocked: list[str] = []
        group_ids: list[str] = []
        seen_paths: set[Path] = set()
        for plan in plans:
            blocked.extend(plan.blocked)
            group_ids.extend(plan.group_ids)
            for item in plan.items:
                if item.path in seen_paths:
                    continue
                seen_paths.add(item.path)
                items.append(item)
        return self._finalize_plan(items, blocked, list(dict.fromkeys(group_ids)))

    def execute_plan(self, plan: DeletionPlan) -> DeletionResult:
        result = DeletionResult()
        # Delete deepest paths first so files go before parent dirs when both listed.
        ordered = sorted(plan.items, key=lambda item: len(item.path.parts), reverse=True)
        removed_dirs: set[Path] = set()

        for item in ordered:
            # Skip if an ancestor dir was already removed.
            if any(path_is_within(item.path, parent) and item.path != parent for parent in removed_dirs):
                continue
            try:
                resolved = assert_path_allowed(item.path, self.allowed_roots)
            except PermissionError as exc:
                result.errors.append(str(exc))
                continue

            if self._is_protected(resolved):
                result.skipped.append(f"Protégé : {resolved}")
                continue

            if not resolved.exists():
                result.skipped.append(f"Déjà absent : {resolved}")
                continue

            size = path_size(resolved)
            try:
                if resolved.is_dir():
                    shutil.rmtree(resolved)
                    result.dirs_removed += 1
                    removed_dirs.add(resolved)
                else:
                    resolved.unlink()
                    result.files_removed += 1
                result.bytes_freed += size
            except OSError as exc:
                result.errors.append(f"{resolved} : {exc}")

            # After deleting source.zip / extracted, prune empty dataset parent.
            self._maybe_prune_empty_dataset_parent(resolved, result)

        return result

    # ------------------------------------------------------------------ builders
    def _build_run_group(self, run_dir: Path) -> StorageGroup:
        status = read_status(run_dir)
        request = read_request_safe(run_dir)
        run_id = (status.run_id if status else None) or run_dir.name
        state = status.state if status else None
        created = status.started_at if status else None
        dataset_id = None
        if request:
            dataset_id = request.get("dataset_id")
            if isinstance(dataset_id, str) and dataset_id.strip():
                dataset_id = dataset_id.strip()
            else:
                dataset_id = None

        dataset_dir = (self.datasets_root / dataset_id) if dataset_id else None
        legacy = status is None and request is None

        title = self._format_run_title(run_id, created)
        subtitle_parts: list[str] = []
        if dataset_id:
            subtitle_parts.append(f"Dataset : {dataset_id}")
        elif request and request.get("dataset_root"):
            subtitle_parts.append(f"Dataset : {Path(str(request['dataset_root'])).name}")
        if state:
            subtitle_parts.append(f"État : {state}")
        if legacy:
            subtitle_parts.append("Ancien entraînement / métadonnées incomplètes")
        subtitle = " · ".join(subtitle_parts) if subtitle_parts else "Session d'entraînement"

        categories = self._classify_run(run_dir)
        shared = False
        if dataset_dir is not None and dataset_dir.is_dir():
            ref_count = self._dataset_reference_count(dataset_id or "")
            shared = ref_count > 1
            categories.extend(self._classify_dataset(dataset_dir, shared=shared))

        categories = [c for c in categories if c.size_bytes > 0 or c.paths]
        size_bytes = sum(c.size_bytes for c in categories)

        protected = False
        reason = None
        if state in ACTIVE_STATES:
            protected = True
            reason = f"Run en état « {state} » — suppression bloquée."
        if self._is_protected(run_dir):
            protected = True
            reason = reason or "Chemin protégé (inférence / session active)."

        return StorageGroup(
            group_id=f"run:{run_id}",
            title=title,
            subtitle=subtitle,
            kind="run",
            run_id=run_id,
            dataset_id=dataset_id,
            run_dir=run_dir.resolve(),
            dataset_dir=dataset_dir.resolve() if dataset_dir and dataset_dir.exists() else None,
            created_at=created,
            state=state,
            categories=categories,
            size_bytes=size_bytes,
            protected=protected,
            protection_reason=reason,
            legacy=legacy,
        )

    def _build_orphan_dataset_group(self, dataset_dir: Path) -> StorageGroup:
        meta = self._read_dataset_meta(dataset_dir)
        dataset_id = dataset_dir.name
        created = meta.get("created_at") if meta else None
        categories = self._classify_dataset(dataset_dir, shared=False)
        size_bytes = sum(c.size_bytes for c in categories)
        protected = self._is_protected(dataset_dir)
        return StorageGroup(
            group_id=f"dataset:{dataset_id}",
            title=f"Dataset orphelin — {dataset_id}",
            subtitle="Non référencé par un run actuel",
            kind="orphan_dataset",
            dataset_id=dataset_id,
            dataset_dir=dataset_dir.resolve(),
            created_at=created if isinstance(created, str) else None,
            categories=categories,
            size_bytes=size_bytes,
            protected=protected,
            protection_reason="Protégé (actif)." if protected else None,
            legacy=meta is None,
        )

    def _classify_run(self, run_dir: Path) -> list[StorageCategory]:
        weights_dir = run_dir / "weights"
        final_paths: list[StoragePathRef] = []
        checkpoint_paths: list[StoragePathRef] = []
        temp_paths: list[StoragePathRef] = []
        artifact_paths: list[StoragePathRef] = []

        if weights_dir.is_dir():
            for path in sorted(weights_dir.iterdir()):
                if not path.is_file():
                    continue
                if self.is_final_model_artifact(path):
                    final_paths.append(StoragePathRef(path=path.resolve(), is_dir=False))
                elif self._is_checkpoint_file(path):
                    checkpoint_paths.append(StoragePathRef(path=path.resolve(), is_dir=False))
                else:
                    artifact_paths.append(StoragePathRef(path=path.resolve(), is_dir=False))

        for path in sorted(run_dir.iterdir()):
            if path.name == "weights":
                continue
            if path.name.startswith("."):
                continue
            if self._is_temp_path(path):
                temp_paths.append(
                    StoragePathRef(path=path.resolve(), is_dir=path.is_dir())
                )
            else:
                artifact_paths.append(
                    StoragePathRef(path=path.resolve(), is_dir=path.is_dir())
                )

        categories = [
            StorageCategory(
                kind=CategoryKind.FINAL_MODEL,
                label=CATEGORY_LABELS[CategoryKind.FINAL_MODEL],
                paths=final_paths,
                size_bytes=sum(path_size(ref.path) for ref in final_paths),
                important=True,
            ),
            StorageCategory(
                kind=CategoryKind.RUN_CHECKPOINTS,
                label=CATEGORY_LABELS[CategoryKind.RUN_CHECKPOINTS],
                paths=checkpoint_paths,
                size_bytes=sum(path_size(ref.path) for ref in checkpoint_paths),
            ),
            StorageCategory(
                kind=CategoryKind.TEMPS,
                label=CATEGORY_LABELS[CategoryKind.TEMPS],
                paths=temp_paths,
                size_bytes=sum(path_size(ref.path) for ref in temp_paths),
            ),
            StorageCategory(
                kind=CategoryKind.RUN_ARTIFACTS,
                label=CATEGORY_LABELS[CategoryKind.RUN_ARTIFACTS],
                paths=artifact_paths,
                size_bytes=sum(path_size(ref.path) for ref in artifact_paths),
            ),
        ]
        return categories

    def _classify_dataset(self, dataset_dir: Path, *, shared: bool) -> list[StorageCategory]:
        archive = dataset_dir / "source.zip"
        extracted = dataset_dir / "extracted"
        categories: list[StorageCategory] = []
        if archive.exists():
            categories.append(
                StorageCategory(
                    kind=CategoryKind.DATASET_ARCHIVE,
                    label=CATEGORY_LABELS[CategoryKind.DATASET_ARCHIVE],
                    paths=[StoragePathRef(path=archive.resolve(), is_dir=False)],
                    size_bytes=path_size(archive),
                    shared=shared,
                )
            )
        if extracted.exists():
            categories.append(
                StorageCategory(
                    kind=CategoryKind.DATASET_EXTRACTED,
                    label=CATEGORY_LABELS[CategoryKind.DATASET_EXTRACTED],
                    paths=[StoragePathRef(path=extracted.resolve(), is_dir=True)],
                    size_bytes=path_size(extracted),
                    shared=shared,
                )
            )
        # Other leftovers under dataset_dir (meta.json etc.) → temps/small artifacts
        extras: list[StoragePathRef] = []
        for path in dataset_dir.iterdir():
            if path.name in {"source.zip", "extracted"}:
                continue
            if path.name == "meta.json":
                continue
            extras.append(StoragePathRef(path=path.resolve(), is_dir=path.is_dir()))
        if extras:
            categories.append(
                StorageCategory(
                    kind=CategoryKind.TEMPS,
                    label=CATEGORY_LABELS[CategoryKind.TEMPS],
                    paths=extras,
                    size_bytes=sum(path_size(ref.path) for ref in extras),
                    shared=shared,
                )
            )
        return categories

    # ------------------------------------------------------------------ helpers
    def _finalize_plan(
        self,
        items: list[DeletionItem],
        blocked: list[str],
        group_ids: list[str],
    ) -> DeletionPlan:
        estimated = sum(item.size_bytes for item in items)
        return DeletionPlan(
            items=items,
            estimated_bytes=estimated,
            group_ids=group_ids,
            includes_final_model=any(item.category == CategoryKind.FINAL_MODEL for item in items)
            or any(item.is_dir and self._dir_has_final_model(item.path) for item in items),
            includes_dataset=any(
                item.category
                in {CategoryKind.DATASET_ARCHIVE, CategoryKind.DATASET_EXTRACTED}
                for item in items
            ),
            blocked=blocked,
        )

    def _dir_has_final_model(self, path: Path) -> bool:
        weights = path / "weights"
        try:
            if not weights.is_dir():
                return False
            return any(
                self.is_final_model_artifact(child)
                for child in weights.iterdir()
                if child.is_file()
            )
        except OSError:
            return False

    def _dataset_reference_count(self, dataset_id: str) -> int:
        if not dataset_id or not self.runs_root.is_dir():
            return 0
        count = 0
        for run_dir in self.runs_root.iterdir():
            if not run_dir.is_dir():
                continue
            request = read_request_safe(run_dir)
            if request and request.get("dataset_id") == dataset_id:
                count += 1
        return count

    def _is_protected(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
        except OSError:
            return False
        for protected in self.protected_paths:
            if resolved == protected or path_is_within(resolved, protected) or path_is_within(
                protected, resolved
            ):
                return True
        return False

    def _is_checkpoint_file(self, path: Path) -> bool:
        name = path.name.lower()
        stem = path.stem.lower()
        if self.is_final_model_artifact(path):
            return False
        if stem == "last" or name.startswith("last."):
            return True
        if stem.startswith("epoch") or name.startswith("epoch"):
            return True
        return False

    def _is_temp_path(self, path: Path) -> bool:
        name = path.name
        lowered = name.lower()
        if any(marker in lowered for marker in TEMP_NAME_MARKERS):
            return True
        if lowered.startswith("train_batch") and lowered.endswith(".jpg"):
            return True
        if lowered == "labels.jpg":
            return True
        return False

    def _read_dataset_meta(self, dataset_dir: Path) -> dict | None:
        meta_path = dataset_dir / "meta.json"
        if not meta_path.is_file():
            return None
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _format_run_title(self, run_id: str, created_at: str | None) -> str:
        label_date = None
        if created_at:
            try:
                moment = datetime.fromisoformat(created_at)
                label_date = moment.astimezone().strftime("%d/%m/%Y %H:%M")
            except ValueError:
                label_date = None
        if label_date is None and len(run_id) >= 15 and run_id[8] == "-":
            # YYYYMMDD-HHMMSS-...
            try:
                stamp = datetime.strptime(run_id[:15], "%Y%m%d-%H%M%S")
                label_date = stamp.strftime("%d/%m/%Y %H:%M")
            except ValueError:
                label_date = None
        if label_date:
            return f"Entraînement — {label_date}"
        return f"Entraînement — {run_id}"

    def _estimate_unique_bytes(self, groups: list[StorageGroup]) -> int:
        """Sum sizes without double-counting shared dataset paths."""
        seen: set[Path] = set()
        total = 0
        for group in groups:
            for category in group.categories:
                for ref in category.paths:
                    try:
                        resolved = ref.path.resolve()
                    except OSError:
                        continue
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    total += path_size(resolved)
        return total

    def _maybe_prune_empty_dataset_parent(self, removed: Path, result: DeletionResult) -> None:
        """If source.zip/extracted removed, drop empty dataset_id folder (and meta.json)."""
        try:
            parent = removed.parent
            if not path_is_within(parent, self.datasets_root):
                return
            if parent.resolve() == self.datasets_root.resolve():
                return
            if not parent.is_dir():
                return
            remaining = [p for p in parent.iterdir()]
            # Only meta.json left → remove whole dataset dir.
            if not remaining or all(p.name == "meta.json" for p in remaining):
                assert_path_allowed(parent, self.allowed_roots)
                size = path_size(parent)
                shutil.rmtree(parent)
                result.dirs_removed += 1
                result.bytes_freed += size
        except (OSError, PermissionError) as exc:
            result.errors.append(f"Nettoyage dataset parent : {exc}")


# Re-export helpers useful for UI / tests
__all__ = [
    "StorageManager",
    "format_bytes",
    "FINAL_MODEL_STEMS",
    "FINAL_MODEL_SUFFIXES",
]
