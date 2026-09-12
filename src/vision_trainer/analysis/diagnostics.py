"""Rule-based diagnostics and improvement recommendations (no generative AI)."""

from __future__ import annotations

from vision_trainer.analysis.models import (
    ClassMetricRow,
    ClassifyErrorSample,
    ConfusionPair,
    DiagnosticFinding,
    RunAnalysis,
)


def build_diagnostics(analysis: RunAnalysis) -> tuple[list[DiagnosticFinding], list[str]]:
    findings: list[DiagnosticFinding] = []
    recommendations: list[str] = []

    findings.extend(_from_confusion(analysis.confusion_pairs, analysis.dataset_stats))
    findings.extend(_from_class_metrics(analysis.class_metrics, analysis.task))
    findings.extend(_from_classify_errors(analysis.classify_errors, analysis.dataset_stats))
    findings.extend(_from_imbalance(analysis.dataset_stats))
    findings.extend(_from_overfit_hint(analysis))

    for finding in findings:
        if finding.recommendation and finding.recommendation not in recommendations:
            recommendations.append(finding.recommendation)

    if not findings:
        findings.append(
            DiagnosticFinding(
                code="no_issue_detected",
                severity="info",
                title="Aucun signal fort détecté",
                detail=(
                    "Les métriques et confusions disponibles ne font pas apparaître "
                    "d'anomalie nette. Cela ne garantit pas l'absence de problèmes."
                ),
                recommendation=None,
            )
        )
    return findings, recommendations


def _from_confusion(
    pairs: list[ConfusionPair],
    dataset_stats: dict,
) -> list[DiagnosticFinding]:
    out: list[DiagnosticFinding] = []
    val_pc = ((dataset_stats.get("per_class") or {}).get("val") or {})
    for pair in pairs[:8]:
        if pair.count < 2:
            continue
        support = val_pc.get(pair.true_label)
        if support:
            detail = (
                f"Classe « {pair.true_label} » : {pair.count}/{support} images de "
                f"validation sont prédites « {pair.pred_label} »."
            )
        else:
            detail = (
                f"Confusion « {pair.true_label} » → « {pair.pred_label} » : "
                f"{pair.count} erreur(s) observée(s)."
            )
        out.append(
            DiagnosticFinding(
                code="class_confusion",
                severity="warning" if pair.count >= 5 else "info",
                title=f"Confusion {pair.true_label} → {pair.pred_label}",
                detail=detail,
                recommendation=(
                    f"Recommandation : ajouter davantage d'exemples permettant de "
                    f"distinguer « {pair.true_label} » et « {pair.pred_label} » "
                    f"(angles, éclairage, packaging proches)."
                ),
            )
        )
    return out


def _from_class_metrics(
    rows: list[ClassMetricRow],
    task: str,
) -> list[DiagnosticFinding]:
    out: list[DiagnosticFinding] = []
    if not rows:
        return out

    def metric_of(row: ClassMetricRow) -> float | None:
        if task == "classify":
            return row.accuracy if row.accuracy is not None else row.recall
        if task == "segment":
            return row.mask_map50 if row.mask_map50 is not None else row.map50
        return row.map50 if row.map50 is not None else row.recall

    values = [(row, metric_of(row)) for row in rows]
    known = [(row, val) for row, val in values if val is not None]
    if len(known) < 2:
        return out
    avg = sum(val for _, val in known) / len(known)
    for row, val in known:
        if val < avg - 0.15:
            name = "rappel/mAP" if task != "classify" else "score"
            out.append(
                DiagnosticFinding(
                    code="weak_class",
                    severity="warning",
                    title=f"Classe faible : {row.class_name}",
                    detail=(
                        f"La classe « {row.class_name} » a un {name} de {val:.3f}, "
                        f"nettement sous la moyenne des classes ({avg:.3f})."
                    ),
                    recommendation=(
                        f"Recommandation : vérifier les annotations et ajouter des "
                        f"exemples variés pour « {row.class_name} »."
                    ),
                )
            )
        if row.recall is not None and row.precision is not None and row.recall < row.precision - 0.2:
            out.append(
                DiagnosticFinding(
                    code="low_recall",
                    severity="warning",
                    title=f"Rappel bas : {row.class_name}",
                    detail=(
                        f"Classe « {row.class_name} » : rappel={row.recall:.3f}, "
                        f"précision={row.precision:.3f}. Davantage d'objets semblent manqués."
                    ),
                    recommendation=(
                        f"Recommandation : vérifier les annotations manquantes et "
                        f"enrichir les exemples de « {row.class_name} »."
                    ),
                )
            )
    return out


def _from_classify_errors(
    errors: list[ClassifyErrorSample],
    dataset_stats: dict,
) -> list[DiagnosticFinding]:
    if not errors:
        return []
    low_conf = [e for e in errors if e.confidence < 0.5]
    if not low_conf:
        return []
    return [
        DiagnosticFinding(
            code="low_confidence_errors",
            severity="info",
            title="Erreurs à faible confiance",
            detail=(
                f"{len(low_conf)} erreur(s) de validation ont une confiance Top-1 < 50 %. "
                "Ces images sont de bons candidats pour une revue prioritaire."
            ),
            recommendation=(
                "Recommandation : examiner ces images (qualité, ambiguïté, mauvaises étiquettes)."
            ),
        )
    ]


def _from_imbalance(dataset_stats: dict) -> list[DiagnosticFinding]:
    per_class = (dataset_stats.get("per_class") or {}).get("train") or {}
    if not isinstance(per_class, dict) or len(per_class) < 2:
        return []
    counts = [(name, int(n)) for name, n in per_class.items() if int(n) >= 0]
    if not counts:
        return []
    max_c = max(n for _, n in counts)
    findings: list[DiagnosticFinding] = []
    for name, n in counts:
        if max_c >= 20 and n <= max(3, max_c // 10):
            findings.append(
                DiagnosticFinding(
                    code="class_imbalance",
                    severity="warning",
                    title=f"Classe sous-représentée : {name}",
                    detail=(
                        f"Train : « {name} » a {n} image(s) vs {max_c} pour la classe "
                        f"la plus fréquente."
                    ),
                    recommendation=(
                        f"Recommandation : équilibrer le dataset en ajoutant des exemples "
                        f"de « {name} » (sans certitude que ce soit la seule cause)."
                    ),
                )
            )
    return findings


def _from_overfit_hint(analysis: RunAnalysis) -> list[DiagnosticFinding]:
    """Only when both train and val curves clearly diverge — conservative."""
    train = analysis.curves.get("train_loss")
    val = analysis.curves.get("val_loss")
    if train is None or val is None or len(train.values) < 3 or len(val.values) < 3:
        return []
    # Compare last third averages
    def _avg_tail(values: list[float | None]) -> float | None:
        chunk = [v for v in values[len(values) // 2 :] if v is not None]
        if not chunk:
            return None
        return sum(chunk) / len(chunk)

    t = _avg_tail(train.values)
    v = _avg_tail(val.values)
    if t is None or v is None or t <= 0:
        return []
    if v > t * 1.8 and v - t > 0.2:
        return [
            DiagnosticFinding(
                code="overfit_risk",
                severity="warning",
                title="Écart train / validation sur la loss",
                detail=(
                    f"Loss validation moyenne récente ({v:.3f}) nettement au-dessus "
                    f"de la loss train ({t:.3f}). Possible surapprentissage — à confirmer."
                ),
                recommendation=(
                    "Recommandation : vérifier la diversité du jeu val, réduire le "
                    "sur-ajustement (régularisation / early stop) ou enrichir les données."
                ),
            )
        ]
    return []
