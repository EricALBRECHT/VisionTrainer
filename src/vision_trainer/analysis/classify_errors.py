"""On-demand classification error analysis (cached in analysis.json)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from vision_trainer.analysis.confusion import (
    build_confusion_matrix,
    confusion_pairs_from_errors,
    confusion_pairs_from_matrix,
)
from vision_trainer.analysis.models import ClassifyErrorSample, TopScore
from vision_trainer.analysis.store import load_analysis, save_analysis
from vision_trainer.tasks import normalize_task
from vision_trainer.training.status import read_request_safe

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class ClassifyAnalysisError(Exception):
    """Raised when classification error analysis cannot run."""


PredictFn = Callable[[Path], tuple[str, float, list[tuple[str, float]]]]
"""Return (pred_label, confidence, top_scores[(name, conf), ...])."""


def analyze_classify_errors(
    run_dir: Path,
    *,
    predict_fn: PredictFn,
    force: bool = False,
    max_errors: int | None = None,
) -> list[ClassifyErrorSample]:
    """
    Evaluate validation images and persist misclassifications into analysis.json.

    Does not invent errors from the confusion PNG. ``predict_fn`` is injected
    (real Ultralytics or test mock).
    """
    run_dir = Path(run_dir)
    existing = load_analysis(run_dir)
    if (
        existing is not None
        and existing.classify_errors_computed
        and not force
    ):
        return list(existing.classify_errors)

    request = read_request_safe(run_dir) or {}
    if normalize_task(request.get("task")) != "classify":
        raise ClassifyAnalysisError("L'analyse d'erreurs image est réservée à la classification.")

    data_dir = request.get("data_dir") or request.get("dataset_root")
    if not data_dir:
        raise ClassifyAnalysisError("Chemin dataset absent de request.json.")
    root = Path(str(data_dir))
    if not root.is_dir():
        raise ClassifyAnalysisError(f"Dataset inaccessible : {root}")

    val_dir = _val_dir(root)
    if val_dir is None:
        raise ClassifyAnalysisError("Dossier validation (val/valid/test) introuvable.")

    samples = list(_iter_labeled_images(val_dir))
    if not samples:
        raise ClassifyAnalysisError("Aucune image de validation trouvée.")

    true_all: list[str] = []
    pred_all: list[str] = []
    errors: list[ClassifyErrorSample] = []
    labels = sorted({label for label, _ in samples})

    for true_label, image_path in samples:
        pred_label, confidence, top = predict_fn(image_path)
        true_all.append(true_label)
        pred_all.append(pred_label)
        if pred_label == true_label:
            continue
        top_scores = [TopScore(class_name=n, confidence=float(c)) for n, c in top]
        margin = None
        if len(top_scores) >= 2:
            margin = float(top_scores[0].confidence - top_scores[1].confidence)
        rel = _safe_relative(image_path, root)
        errors.append(
            ClassifyErrorSample(
                image_path=rel,
                true_label=true_label,
                pred_label=pred_label,
                confidence=float(confidence),
                top_scores=top_scores,
                margin_top1_top2=margin,
            )
        )

    errors.sort(key=lambda e: (-e.confidence, e.true_label, e.pred_label))
    if max_errors is not None:
        stored_errors = errors[: max(0, int(max_errors))]
    else:
        stored_errors = errors

    matrix = build_confusion_matrix(labels, true_labels=true_all, pred_labels=pred_all)
    pairs = confusion_pairs_from_matrix(matrix, labels)
    if not pairs:
        pairs = confusion_pairs_from_errors(errors)

    from vision_trainer.analysis.builder import build_run_analysis

    analysis = build_run_analysis(run_dir, persist=False)
    analysis.classify_errors = stored_errors
    analysis.classify_errors_computed = True
    analysis.confusion_matrix = matrix
    analysis.confusion_labels = labels
    analysis.confusion_pairs = pairs
    # Refresh diagnostics with new confusion/errors
    from vision_trainer.analysis.diagnostics import build_diagnostics

    analysis.diagnostics, analysis.recommendations = build_diagnostics(analysis)
    save_analysis(run_dir, analysis)
    return stored_errors


def default_ultralytics_predict_fn(
    weights_path: Path,
    *,
    device: str = "cpu",
    top_n: int = 3,
    model_factory: Callable[[str], Any] | None = None,
) -> PredictFn:
    """Build a PredictFn backed by Ultralytics YOLO-cls (lazy import)."""

    factory = model_factory
    model_holder: dict[str, Any] = {}

    def _predict(image_path: Path) -> tuple[str, float, list[tuple[str, float]]]:
        if "model" not in model_holder:
            if factory is None:
                from ultralytics import YOLO

                model_holder["model"] = YOLO(str(weights_path))
            else:
                model_holder["model"] = factory(str(weights_path))
        model = model_holder["model"]
        results = model.predict(source=str(image_path), device=device, verbose=False)
        if not results:
            return ("", 0.0, [])
        first = results[0]
        probs = getattr(first, "probs", None)
        if probs is None:
            return ("", 0.0, [])
        names = getattr(first, "names", None) or getattr(model, "names", None) or {}
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        data = getattr(probs, "data", None)
        if data is None:
            return ("", 0.0, [])
        values = data.tolist() if hasattr(data, "tolist") else list(data)
        ranked = sorted(
            (
                (
                    str(names.get(i, i)) if isinstance(names, dict) else str(i),
                    float(conf),
                )
                for i, conf in enumerate(values)
            ),
            key=lambda x: -x[1],
        )
        if not ranked:
            return ("", 0.0, [])
        top = ranked[:top_n]
        return top[0][0], top[0][1], top

    return _predict


def _val_dir(root: Path) -> Path | None:
    for name in ("val", "valid", "test"):
        path = root / name
        if path.is_dir():
            return path
    return None


def _iter_labeled_images(val_dir: Path) -> list[tuple[str, Path]]:
    samples: list[tuple[str, Path]] = []
    for class_dir in sorted(val_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        label = class_dir.name
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append((label, path))
    return samples


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name
