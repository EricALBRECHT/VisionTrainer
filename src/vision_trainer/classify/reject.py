"""Confidence rejection helpers for classification inference (V1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DecisionKind = Literal["accepted", "unknown", "uncertain"]


@dataclass(frozen=True)
class ClassScore:
    class_id: int
    class_name: str
    confidence: float


@dataclass(frozen=True)
class ClassifyDecision:
    kind: DecisionKind
    label: str
    top1: ClassScore | None
    top2: ClassScore | None
    top_n: tuple[ClassScore, ...]
    reason: str | None = None


def decide_classification(
    scores: list[ClassScore],
    *,
    min_confidence: float = 0.80,
    min_margin: float = 0.10,
    top_n: int = 5,
) -> ClassifyDecision:
    """
    Apply simple Top-1 rejection rules.

    - ``unknown`` when Top-1 confidence < ``min_confidence``
    - ``uncertain`` when Top-1 − Top-2 < ``min_margin`` (even if above threshold)
    - otherwise ``accepted`` with the Top-1 class

    This is **not** a formal out-of-distribution detector.
    """
    ordered = tuple(sorted(scores, key=lambda item: item.confidence, reverse=True))
    shown = ordered[: max(1, int(top_n))]
    if not ordered:
        return ClassifyDecision(
            kind="unknown",
            label="INCONNU / CONFIANCE INSUFFISANTE",
            top1=None,
            top2=None,
            top_n=(),
            reason="Aucune probabilité disponible.",
        )

    top1 = ordered[0]
    top2 = ordered[1] if len(ordered) > 1 else None

    if top1.confidence < float(min_confidence):
        return ClassifyDecision(
            kind="unknown",
            label="INCONNU / CONFIANCE INSUFFISANTE",
            top1=top1,
            top2=top2,
            top_n=shown,
            reason=(
                f"Score Top-1 ({top1.confidence:.2f}) inférieur au seuil "
                f"minimum ({float(min_confidence):.2f})."
            ),
        )

    if top2 is not None and (top1.confidence - top2.confidence) < float(min_margin):
        return ClassifyDecision(
            kind="uncertain",
            label="INCERTAIN",
            top1=top1,
            top2=top2,
            top_n=shown,
            reason=(
                f"Écart Top-1/Top-2 trop faible "
                f"({top1.confidence:.2f} − {top2.confidence:.2f} "
                f"< {float(min_margin):.2f})."
            ),
        )

    return ClassifyDecision(
        kind="accepted",
        label=top1.class_name,
        top1=top1,
        top2=top2,
        top_n=shown,
        reason=None,
    )
