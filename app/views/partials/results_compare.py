"""Streamlit: Comparaison de deux runs (Résultats → Comparaison)."""

from __future__ import annotations

import json

import streamlit as st

from vision_trainer.analysis.compare import compare_runs, format_pp
from vision_trainer.results.catalog import discover_runs, format_duration, format_optional
from vision_trainer.tasks import task_label_fr
from vision_trainer.training.runs import ARTIFACTS_RUNS_DIR


def render() -> None:
    st.markdown(
        "Comparez deux entraînements de **même type** "
        "(détection / classification / segmentation)."
    )
    runs = discover_runs(ARTIFACTS_RUNS_DIR)
    if len(runs) < 2:
        st.info("Il faut au moins deux runs pour comparer.")
        return

    labels = [
        f"{r.run_id} — {task_label_fr(r.task)} — {r.state}"
        for r in runs
    ]
    col_a, col_b = st.columns(2)
    with col_a:
        label_a = st.selectbox("Run A", options=labels, key="cmp_run_a")
    with col_b:
        label_b = st.selectbox(
            "Run B",
            options=labels,
            index=min(1, len(labels) - 1),
            key="cmp_run_b",
        )

    run_a = runs[labels.index(label_a)]
    run_b = runs[labels.index(label_b)]
    if run_a.run_id == run_b.run_id:
        st.warning("Sélectionnez deux runs distincts.")
        return

    if st.button("Comparer", type="primary", key="cmp_go"):
        comparison = compare_runs(run_a.run_dir, run_b.run_dir)
        st.session_state["last_run_comparison"] = comparison.to_dict()

    payload = st.session_state.get("last_run_comparison")
    if not payload:
        return

    # Re-bind ids in case selection changed without re-compare
    if payload.get("run_a_id") != run_a.run_id or payload.get("run_b_id") != run_b.run_id:
        st.info("Cliquez sur **Comparer** pour actualiser avec la sélection courante.")
        return

    from vision_trainer.analysis.models import (
        ClassDelta,
        MetricDelta,
        RunComparison,
    )

    comparison = RunComparison(
        comparison_version=int(payload.get("comparison_version") or 1),
        task=str(payload.get("task") or "detect"),
        run_a_id=str(payload.get("run_a_id") or ""),
        run_b_id=str(payload.get("run_b_id") or ""),
        compatible=bool(payload.get("compatible", True)),
        warnings=[str(w) for w in (payload.get("warnings") or [])],
        summary_a=dict(payload.get("summary_a") or {}),
        summary_b=dict(payload.get("summary_b") or {}),
        metric_deltas=[
            MetricDelta(**m) for m in (payload.get("metric_deltas") or []) if isinstance(m, dict)
        ],
        class_deltas=[
            ClassDelta(**c) for c in (payload.get("class_deltas") or []) if isinstance(c, dict)
        ],
        regressions=[
            ClassDelta(**c) for c in (payload.get("regressions") or []) if isinstance(c, dict)
        ],
        improvements=[
            ClassDelta(**c) for c in (payload.get("improvements") or []) if isinstance(c, dict)
        ],
        duration_a_seconds=payload.get("duration_a_seconds"),
        duration_b_seconds=payload.get("duration_b_seconds"),
        speedup_b_vs_a=payload.get("speedup_b_vs_a"),
    )

    for warning in comparison.warnings:
        st.warning(warning)

    if not comparison.compatible:
        st.error("Comparaison métrique non applicable (tâches différentes).")
        return

    st.subheader("Configuration")
    sa, sb = comparison.summary_a, comparison.summary_b
    st.dataframe(
        [
            {
                "champ": "Dataset",
                "Run A": format_optional(sa.get("dataset_name")),
                "Run B": format_optional(sb.get("dataset_name")),
            },
            {
                "champ": "Modèle",
                "Run A": format_optional(sa.get("model")),
                "Run B": format_optional(sb.get("model")),
            },
            {
                "champ": "Epochs",
                "Run A": format_optional(sa.get("epochs")),
                "Run B": format_optional(sb.get("epochs")),
            },
            {
                "champ": "imgsz",
                "Run A": format_optional(sa.get("imgsz")),
                "Run B": format_optional(sb.get("imgsz")),
            },
            {
                "champ": "batch",
                "Run A": format_optional(sa.get("batch")),
                "Run B": format_optional(sb.get("batch")),
            },
            {
                "champ": "device",
                "Run A": format_optional(sa.get("device_name") or sa.get("device")),
                "Run B": format_optional(sb.get("device_name") or sb.get("device")),
            },
            {
                "champ": "durée",
                "Run A": format_duration(comparison.duration_a_seconds),
                "Run B": format_duration(comparison.duration_b_seconds),
            },
        ],
        use_container_width=True,
        hide_index=True,
    )

    if comparison.speedup_b_vs_a is not None:
        st.caption(
            f"Run B est environ **×{comparison.speedup_b_vs_a}** plus rapide que Run A "
            "(dépend aussi du hardware, batch, imgsz et dataset)."
        )

    st.subheader("Métriques globales")
    for delta in comparison.metric_deltas:
        st.write(
            f"- **{delta.name}** : "
            f"{format_optional(delta.value_a, percent=True)} → "
            f"{format_optional(delta.value_b, percent=True)} "
            f"({format_pp(delta.delta_pp)})"
        )

    if comparison.regressions:
        st.subheader("Régressions (par classe)")
        st.caption("Amélioration globale possible malgré des classes en baisse.")
        for row in comparison.regressions:
            st.write(
                f"- **{row.class_name}** ({row.metric_name}) : "
                f"{format_optional(row.value_a, percent=True)} → "
                f"{format_optional(row.value_b, percent=True)} "
                f"({format_pp(row.delta_pp)})"
            )

    if comparison.improvements:
        with st.expander("Améliorations par classe", expanded=False):
            for row in comparison.improvements:
                st.write(
                    f"- **{row.class_name}** : {format_pp(row.delta_pp)}"
                )

    if comparison.class_deltas:
        with st.expander("Tous les deltas par classe", expanded=False):
            st.dataframe(
                [c.to_dict() for c in comparison.class_deltas],
                use_container_width=True,
                hide_index=True,
            )
    elif comparison.compatible:
        st.caption(
            "Pas de métriques par classe numériques comparables pour ces runs "
            "(souvent le cas avec les exports Ultralytics standards)."
        )

    st.download_button(
        "Exporter comparison.json",
        data=json.dumps(comparison.to_dict(), indent=2, ensure_ascii=False),
        file_name=f"comparison_{comparison.run_a_id}_vs_{comparison.run_b_id}.json",
        mime="application/json",
        key="dl_comparison",
    )
