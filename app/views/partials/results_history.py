from __future__ import annotations

from pathlib import Path

import streamlit as st

from vision_trainer.results.catalog import (
    discover_runs,
    format_duration,
    format_optional,
    load_metrics_history,
    load_run,
)
from vision_trainer.results.models import SESSION_INFERENCE_WEIGHTS_KEY
from vision_trainer.tasks import task_label_fr
from vision_trainer.training.runs import ARTIFACTS_RUNS_DIR

from navigation import PAGE_INFERENCE


def render(*, embedded: bool = False) -> None:
    if not embedded:
        st.title("Résultats")
    st.markdown(
        "Consultez les entraînements (détection, classification, segmentation) "
        "dans les runs."
    )

    runs = discover_runs(ARTIFACTS_RUNS_DIR)
    if not runs:
        st.info(
            "Aucun run trouvé. Lancez un entraînement depuis la rubrique "
            "**Entraînement** (détection, classification ou segmentation)."
        )
        st.stop()

    filter_label = st.segmented_control(
        "Filtrer par tâche",
        options=["Tous", "Détection", "Classification", "Segmentation"],
        default="Tous",
        key="results_task_filter",
        required=True,
    )
    task_filter = {
        "Tous": None,
        "Détection": "detect",
        "Classification": "classify",
        "Segmentation": "segment",
    }.get(filter_label or "Tous")
    if task_filter is not None:
        runs = [item for item in runs if item.task == task_filter]
        if not runs:
            st.info(f"Aucun run de type **{filter_label}** pour le moment.")
            st.stop()

    labels = []
    for summary in runs:
        date_label = summary.started_at or summary.finished_at or "date inconnue"
        if summary.started_at and "T" in summary.started_at:
            date_label = summary.started_at.split("T", 1)[0]
        labels.append(
            f"{summary.run_id} — {task_label_fr(summary.task)} — {summary.state} — {date_label}"
        )

    st.subheader("Entraînements")
    selected_label = st.selectbox("Sélectionner un run", options=labels)
    selected_summary = runs[labels.index(selected_label)]

    # Compact list overview
    with st.expander("Liste compacte de tous les runs", expanded=False):
        st.dataframe(
            [
                {
                    "run_id": item.run_id,
                    "type": task_label_fr(item.task),
                    "état": item.state,
                    "date": (item.started_at or item.finished_at or "Non disponible"),
                    "durée": format_duration(item.duration_seconds),
                    "modèle": format_optional(item.model),
                    "epochs": format_optional(item.epochs),
                    "imgsz": format_optional(item.imgsz),
                    "batch": format_optional(item.batch),
                    "device": format_optional(item.device),
                    "classes": format_optional(item.num_classes),
                    "best.pt": "oui" if item.has_best else "non",
                }
                for item in runs
            ],
            use_container_width=True,
            hide_index=True,
        )

    detail = load_run(selected_summary.run_dir)
    summary = detail.summary

    st.subheader(f"Détail — `{summary.run_id}`")
    st.caption(f"Dossier : `{summary.run_dir}`")
    if summary.load_warning:
        st.warning(summary.load_warning)
    if detail.error_message and summary.state in {"failed", "interrupted"}:
        st.error(detail.error_message)

    col_cfg, col_metrics, col_weights = st.columns(3)

    with col_cfg:
        st.markdown("#### Configuration")
        st.write(f"- Type : **{task_label_fr(summary.task)}**")
        st.write(f"- Modèle : `{format_optional(summary.model)}`")
        st.write(f"- Epochs : {format_optional(summary.epochs)}")
        st.write(f"- imgsz : {format_optional(summary.imgsz)}")
        st.write(f"- batch : {format_optional(summary.batch)}")
        st.write(f"- device : {format_optional(summary.device)}")
        st.write(f"- Classes : {format_optional(summary.num_classes)}")
        st.write(f"- État : **{summary.state}**")
        st.write(f"- Durée : {format_duration(summary.duration_seconds)}")

    with col_metrics:
        st.markdown("#### Résultats")
        if summary.task == "classify":
            st.write(f"- Accuracy Top-1 : {format_optional(detail.accuracy_top1, percent=True)}")
            st.write(f"- Accuracy Top-5 : {format_optional(detail.accuracy_top5, percent=True)}")
        elif summary.task == "segment":
            st.markdown("**Box**")
            st.write(f"- Precision : {format_optional(detail.precision, percent=True)}")
            st.write(f"- Recall : {format_optional(detail.recall, percent=True)}")
            st.write(f"- mAP50 : {format_optional(detail.map50, percent=True)}")
            st.write(f"- mAP50-95 : {format_optional(detail.map50_95, percent=True)}")
            st.markdown("**Mask**")
            st.write(f"- Precision : {format_optional(detail.mask_precision, percent=True)}")
            st.write(f"- Recall : {format_optional(detail.mask_recall, percent=True)}")
            st.write(f"- mAP50 : {format_optional(detail.mask_map50, percent=True)}")
            st.write(f"- mAP50-95 : {format_optional(detail.mask_map50_95, percent=True)}")
        else:
            st.write(f"- Precision : {format_optional(detail.precision, percent=True)}")
            st.write(f"- Recall : {format_optional(detail.recall, percent=True)}")
            st.write(f"- mAP50 : {format_optional(detail.map50, percent=True)}")
            st.write(f"- mAP50-95 : {format_optional(detail.map50_95, percent=True)}")

    with col_weights:
        st.markdown("#### Modèles")
        st.write(f"- best.pt : `{detail.best_pt if detail.best_pt else 'Non disponible'}`")
        st.write(f"- last.pt : `{detail.last_pt if detail.last_pt else 'Non disponible'}`")

    if detail.best_pt is not None and summary.state in {"terminé", "completed", "interrupted"}:
        if st.button("Utiliser ce modèle pour l'inférence", type="primary"):
            st.session_state[SESSION_INFERENCE_WEIGHTS_KEY] = str(detail.best_pt.resolve())
            try:
                st.session_state["hub_infer_task"] = {
                    "detect": "Détection",
                    "classify": "Classification",
                    "segment": "Segmentation",
                }.get(summary.task, "Détection")
                st.switch_page(PAGE_INFERENCE)
            except Exception:  # noqa: BLE001 - older Streamlit fallback
                st.success(
                    "Modèle sélectionné pour l'inférence. Ouvrez la rubrique **Inférence**."
                )

    history = load_metrics_history(summary.run_dir)
    if summary.task == "classify":
        st.caption("Métriques détection/segmentation (mAP) non applicables à la classification.")
    elif history is not None and (
        history.has_map50
        or history.has_map50_95
        or history.has_mask_map50
        or history.has_mask_map50_95
    ):
        st.subheader("Courbes d'entraînement (results.csv)")
        rows: list[dict] = []
        for index, epoch in enumerate(history.epochs):
            row: dict = {"epoch": epoch}
            if history.has_map50 and history.map50[index] is not None:
                row["Box mAP50"] = history.map50[index]
            if history.has_map50_95 and history.map50_95[index] is not None:
                row["Box mAP50-95"] = history.map50_95[index]
            if history.has_mask_map50 and history.mask_map50[index] is not None:
                row["Mask mAP50"] = history.mask_map50[index]
            if history.has_mask_map50_95 and history.mask_map50_95[index] is not None:
                row["Mask mAP50-95"] = history.mask_map50_95[index]
            # Keep legacy keys for detect-only charts readability
            if summary.task != "segment":
                if "Box mAP50" in row:
                    row["mAP50"] = row.pop("Box mAP50")
                if "Box mAP50-95" in row:
                    row["mAP50-95"] = row.pop("Box mAP50-95")
            rows.append(row)
        if summary.task == "segment":
            y_cols = [
                key
                for key in ("Box mAP50", "Box mAP50-95", "Mask mAP50", "Mask mAP50-95")
                if any(key in row for row in rows)
            ]
        else:
            y_cols = [key for key in ("mAP50", "mAP50-95") if any(key in row for row in rows)]
        if y_cols:
            st.line_chart(rows, x="epoch", y=y_cols)
    elif history is None:
        st.caption("Pas de `results.csv` pour ce run.")
    else:
        st.caption("`results.csv` présent mais sans colonnes mAP détectées.")

    st.subheader("Graphiques Ultralytics")
    if not detail.plots:
        st.caption("Aucun graphique Ultralytics trouvé dans ce dossier.")
    else:
        # Display up to 2 per row
        plot_items = list(detail.plots.items())
        for index in range(0, len(plot_items), 2):
            cols = st.columns(2)
            for offset, (name, path) in enumerate(plot_items[index : index + 2]):
                with cols[offset]:
                    st.markdown(f"**{name}**")
                    try:
                        st.image(str(path), use_container_width=True)
                    except Exception:  # noqa: BLE001
                        st.warning(f"Impossible d'afficher `{name}`.")
