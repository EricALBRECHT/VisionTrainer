"""Streamlit: Analyse d'un run (Résultats → Analyse)."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from vision_trainer.analysis.builder import build_run_analysis
from vision_trainer.analysis.classify_errors import (
    ClassifyAnalysisError,
    analyze_classify_errors,
    default_ultralytics_predict_fn,
)
from vision_trainer.analysis.models import RunAnalysis
from vision_trainer.analysis.store import analysis_path, load_analysis
from vision_trainer.results.catalog import discover_runs, format_optional
from vision_trainer.tasks import task_label_fr
from vision_trainer.training.runs import ARTIFACTS_RUNS_DIR


DEFAULT_ERROR_PREVIEW = 20


def render() -> None:
    st.markdown(
        "Analysez un entraînement : métriques, confusions, erreurs "
        "(classification), diagnostics et pistes d'amélioration."
    )
    runs = discover_runs(ARTIFACTS_RUNS_DIR)
    if not runs:
        st.info("Aucun run disponible.")
        return

    labels = [
        f"{r.run_id} — {task_label_fr(r.task)} — {r.state}"
        for r in runs
    ]
    preferred_id = st.session_state.get("analysis_preferred_run_id")
    default_index = 0
    if preferred_id:
        for idx, item in enumerate(runs):
            if item.run_id == preferred_id:
                default_index = idx
                break
    selected = st.selectbox(
        "Run à analyser",
        options=labels,
        index=default_index,
        key="analysis_run_select",
    )
    summary = runs[labels.index(selected)]
    run_dir = summary.run_dir

    col_a, col_b = st.columns(2)
    with col_a:
        refresh = st.button("Analyser / Actualiser", type="primary", key="analysis_build")
    with col_b:
        force_stats = st.checkbox(
            "Rescanner stats dataset",
            value=False,
            key="analysis_force_stats",
        )

    analysis: RunAnalysis | None = load_analysis(run_dir)
    if refresh or analysis is None:
        with st.spinner("Construction de l'analyse…"):
            analysis = build_run_analysis(
                run_dir,
                persist=True,
                refresh_dataset_stats=force_stats,
            )

    assert analysis is not None
    _render_analysis(analysis, run_dir)


def _render_analysis(analysis: RunAnalysis, run_dir: Path) -> None:
    s = analysis.summary
    st.subheader("Résumé")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(f"- **Dataset** : `{format_optional(s.get('dataset_name'))}`")
        st.write(f"- **Type** : {s.get('task_label') or task_label_fr(analysis.task)}")
        st.write(f"- **Modèle** : `{format_optional(s.get('model'))}`")
        st.write(f"- **Date** : {format_optional(s.get('started_at'))}")
    with c2:
        st.write(f"- **Epochs** : {format_optional(s.get('epochs'))}")
        st.write(f"- **imgsz** : {format_optional(s.get('imgsz'))}")
        st.write(f"- **batch** : {format_optional(s.get('batch'))}")
        st.write(f"- **Durée** : {format_optional(s.get('duration_label'))}")
    with c3:
        st.write(f"- **Device** : `{format_optional(s.get('device'))}`")
        st.write(f"- **Matériel** : {format_optional(s.get('device_name'))}")
        st.write(f"- **Seed** : {format_optional(s.get('seed'))}")
        export_name = s.get("export_model_name")
        if export_name:
            st.write(f"- **Export** : `{export_name}`")

    ds = analysis.dataset_stats or {}
    if ds:
        st.caption(
            f"Train : {format_optional(ds.get('train_images'))} images — "
            f"Val : {format_optional(ds.get('val_images'))} images"
        )

    st.subheader("Métriques principales")
    _render_metrics(analysis)

    if analysis.class_metrics:
        st.subheader("Métriques par classe")
        st.dataframe(
            [row.to_dict() for row in analysis.class_metrics],
            use_container_width=True,
            hide_index=True,
        )

    if analysis.confusion_pairs:
        st.subheader("Confusions principales")
        for pair in analysis.confusion_pairs[:15]:
            st.write(
                f"- **{pair.true_label}** → **{pair.pred_label}** : "
                f"{pair.count} erreur(s)"
            )

    if analysis.task == "classify":
        _render_classify_errors_section(analysis, run_dir)

    if analysis.task in {"detect", "segment"} and analysis.detection_errors_v1_note:
        with st.expander("Erreurs détection / segmentation (V1)", expanded=False):
            st.info(analysis.detection_errors_v1_note)

    st.subheader("Diagnostic")
    for finding in analysis.diagnostics:
        icon = {"critical": "error", "warning": "warning"}.get(finding.severity, "info")
        getattr(st, icon)(f"**{finding.title}** — {finding.detail}")

    if analysis.recommendations:
        st.subheader("Pistes d'amélioration")
        for rec in analysis.recommendations:
            st.write(f"- {rec}")

    if analysis.curves:
        st.subheader("Courbes d'entraînement")
        _render_curves(analysis)

    if analysis.best_epoch is not None:
        st.caption(
            f"Meilleur epoch estimé : **{analysis.best_epoch}** "
            f"({analysis.best_metric_name} = {analysis.best_metric_value})"
        )

    useful_assets = [
        name
        for name in (
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "BoxPR_curve.png",
            "PR_curve.png",
            "BoxF1_curve.png",
            "F1_curve.png",
            "results.png",
            "MaskPR_curve.png",
            "MaskF1_curve.png",
        )
        if name in analysis.assets
    ]
    if useful_assets:
        with st.expander("Graphiques Ultralytics utiles", expanded=False):
            for name in useful_assets:
                st.markdown(f"**{name}**")
                try:
                    st.image(analysis.assets[name], use_container_width=True)
                except Exception:  # noqa: BLE001
                    st.caption(f"Impossible d'afficher `{name}`.")

    with st.expander("Détails techniques", expanded=False):
        st.write(f"- analysis_version : `{analysis.analysis_version}`")
        st.write(f"- Fichier : `{analysis_path(run_dir)}`")
        env = s.get("environment") or {}
        if env:
            st.json(env)
        if analysis.limitations:
            st.markdown("**Limitations**")
            for line in analysis.limitations:
                st.write(f"- {line}")
        st.download_button(
            "Exporter analysis.json",
            data=json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False),
            file_name=f"analysis_{analysis.run_id}.json",
            mime="application/json",
            key=f"dl_analysis_{analysis.run_id}",
        )


def _render_metrics(analysis: RunAnalysis) -> None:
    m = analysis.metrics
    if analysis.task == "classify":
        st.write(f"- Top-1 : {format_optional(m.get('accuracy_top1'), percent=True)}")
        st.write(f"- Top-5 : {format_optional(m.get('accuracy_top5'), percent=True)}")
        return
    if analysis.task == "segment":
        st.markdown("**BOX**")
        st.write(f"- Precision : {format_optional(m.get('precision'), percent=True)}")
        st.write(f"- Recall : {format_optional(m.get('recall'), percent=True)}")
        st.write(f"- mAP50 : {format_optional(m.get('map50'), percent=True)}")
        st.write(f"- mAP50-95 : {format_optional(m.get('map50_95'), percent=True)}")
        st.markdown("**MASK**")
        st.write(f"- Precision : {format_optional(m.get('mask_precision'), percent=True)}")
        st.write(f"- Recall : {format_optional(m.get('mask_recall'), percent=True)}")
        st.write(f"- mAP50 : {format_optional(m.get('mask_map50'), percent=True)}")
        st.write(f"- mAP50-95 : {format_optional(m.get('mask_map50_95'), percent=True)}")
        return
    st.write(f"- Precision : {format_optional(m.get('precision'), percent=True)}")
    st.write(f"- Recall : {format_optional(m.get('recall'), percent=True)}")
    st.write(f"- mAP50 : {format_optional(m.get('map50'), percent=True)}")
    st.write(f"- mAP50-95 : {format_optional(m.get('map50_95'), percent=True)}")


def _render_curves(analysis: RunAnalysis) -> None:
    preferred = [
        "accuracy_top1",
        "map50",
        "mask_map50",
        "train_loss",
        "val_loss",
    ]
    rows: list[dict] = []
    # Align on union of epochs from first available series
    base = None
    for key in preferred:
        if key in analysis.curves:
            base = analysis.curves[key]
            break
    if base is None:
        base = next(iter(analysis.curves.values()))
    for idx, epoch in enumerate(base.epochs):
        row: dict = {"epoch": epoch}
        for key in preferred:
            series = analysis.curves.get(key)
            if series is None or idx >= len(series.values):
                continue
            val = series.values[idx]
            if val is not None:
                row[series.label or key] = val
        rows.append(row)
    y_cols = [k for k in rows[0].keys() if k != "epoch"] if rows else []
    if y_cols:
        st.line_chart(rows, x="epoch", y=y_cols)


def _render_classify_errors_section(analysis: RunAnalysis, run_dir: Path) -> None:
    st.subheader("Erreurs de classification")
    if not analysis.classify_errors_computed:
        st.caption(
            "Les erreurs image par image ne sont pas inventées depuis la matrice PNG. "
            "Lancez une évaluation du best.pt sur le jeu validation (mis en cache)."
        )
        if st.button("Analyser les erreurs", key="analysis_run_errors"):
            best = run_dir / "weights" / "best.pt"
            if not best.is_file():
                st.error("weights/best.pt introuvable.")
                return
            try:
                with st.spinner("Évaluation validation…"):
                    predict_fn = default_ultralytics_predict_fn(best)
                    analyze_classify_errors(run_dir, predict_fn=predict_fn)
                st.success("Analyse des erreurs enregistrée.")
                st.rerun()
            except ClassifyAnalysisError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Échec de l'analyse : {exc}")
        return

    errors = analysis.classify_errors
    if not errors:
        st.success("Aucune erreur de classification sur le jeu de validation analysé.")
        return

    limit = st.slider(
        "Nombre d'erreurs affichées",
        min_value=5,
        max_value=max(20, min(100, len(errors))),
        value=min(DEFAULT_ERROR_PREVIEW, len(errors)),
        key="analysis_error_limit",
    )
    root = analysis.summary.get("dataset_root")
    root_path = Path(str(root)) if root else None
    for err in errors[:limit]:
        cols = st.columns([1, 2])
        with cols[0]:
            image_path = Path(err.image_path)
            if root_path and not image_path.is_file():
                candidate = root_path / err.image_path
                if candidate.is_file():
                    image_path = candidate
            if image_path.is_file():
                try:
                    st.image(str(image_path), use_container_width=True)
                except Exception:  # noqa: BLE001
                    st.caption(err.image_path)
            else:
                st.caption(err.image_path)
        with cols[1]:
            st.write(f"**Réel** : {err.true_label}")
            st.write(f"**Prédit** : {err.pred_label}")
            st.write(f"**Confiance** : {err.confidence:.0%}")
            if err.top_scores:
                st.markdown("Top-N")
                for score in err.top_scores[:3]:
                    st.write(f"- {score.class_name} — {score.confidence:.0%}")
