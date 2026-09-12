"""Confusion matrix helpers (numeric, not PNG-only)."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from vision_trainer.analysis.models import ClassifyErrorSample, ConfusionPair


def confusion_pairs_from_matrix(
    matrix: list[list[int]],
    labels: list[str],
) -> list[ConfusionPair]:
    """Off-diagonal pairs sorted by count descending."""
    pairs: list[ConfusionPair] = []
    size = min(len(matrix), len(labels))
    for i in range(size):
        row = matrix[i] if i < len(matrix) else []
        for j in range(min(len(row), size)):
            if i == j:
                continue
            count = int(row[j] or 0)
            if count > 0:
                pairs.append(
                    ConfusionPair(
                        true_label=labels[i],
                        pred_label=labels[j],
                        count=count,
                    )
                )
    pairs.sort(key=lambda p: (-p.count, p.true_label, p.pred_label))
    return pairs


def confusion_pairs_from_errors(
    errors: Iterable[ClassifyErrorSample],
) -> list[ConfusionPair]:
    counter: Counter[tuple[str, str]] = Counter()
    for err in errors:
        if err.true_label == err.pred_label:
            continue
        counter[(err.true_label, err.pred_label)] += 1
    pairs = [
        ConfusionPair(true_label=a, pred_label=b, count=n)
        for (a, b), n in counter.items()
    ]
    pairs.sort(key=lambda p: (-p.count, p.true_label, p.pred_label))
    return pairs


def build_confusion_matrix(
    labels: list[str],
    *,
    true_labels: list[str],
    pred_labels: list[str],
) -> list[list[int]]:
    index = {name: i for i, name in enumerate(labels)}
    size = len(labels)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for true, pred in zip(true_labels, pred_labels):
        if true not in index or pred not in index:
            continue
        matrix[index[true]][index[pred]] += 1
    return matrix
