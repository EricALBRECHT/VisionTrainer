from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from vision_trainer.classify.session import (
    SESSION_CLS_DATASET_KEY,
    classify_dataset_from_session_payload,
)
from vision_trainer.tasks import normalize_task, task_label_fr
from vision_trainer.training.classify_trainer import (
    AVAILABLE_CLS_MODELS,
    CLS_IMGSZ_CHOICES,
    DEFAULT_CLS_IMGSZ,
    DEFAULT_CLS_MODEL_KEY,
    SESSION_ACTIVE_CLS_RUN_KEY,
    ClassifyTrainingRequest,
    prepare_classify_training_run,
)
from vision_trainer.training.device import describe_device
from vision_trainer.training.runs import ARTIFACTS_RUNS_DIR
from vision_trainer.training.status import (
    ACTIVE_STATES,
    TRAINING_UI_REFRESH_SECONDS,
    find_reserved_run,
    read_log_tail,
    read_request_safe,
    should_auto_refresh,
)
from vision_trainer.training.trainer import (
    BATCH_AUTO,
    DEFAULT_EPOCHS,
    TrainingError,
    load_run_for_ui,
    start_training_subprocess,
)
from vision_trainer.ui.device_selector import render_device_selector
from views.partials.train_log_display import show_training_log


def render() -> None:
    st.subheader("Classification")
    st.markdown(
        "Configurez et lancez un entraînement **YOLO-cls** (Ultralytics). "
        "Indépendant de la détection d'objets."
    )


    def _parse_iso(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None


    def _format_duration(started_at: str | None, finished_at: str | None) -> str:
        start = _parse_iso(started_at)
        if start is None:
            return "—"
        end = _parse_iso(finished_at) or datetime.now(timezone.utc)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        seconds = max(0, int((end - start).total_seconds()))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours}h {minutes:02d}m {secs:02d}s"
        return f"{minutes}m {secs:02d}s"


    def _format_metric(value: float | None) -> str:
        if value is None:
            return "—"
        return f"{value:.4f}"


    def _batch_label(batch: int) -> str:
        return "auto" if batch == BATCH_AUTO else str(batch)


    def _run_task(run_dir: Path, status) -> str:
        req = read_request_safe(run_dir)
        if req and req.get("task"):
            return normalize_task(req.get("task"))
        return normalize_task(getattr(status, "task", None))


    def _render_run_followup(status, run_dir: Path) -> None:
        st.subheader("Suivi de l'entraînement")
        st.caption(
            f"Run : `{status.run_id}` — **{task_label_fr(status.task)}** — dossier `{run_dir}`"
        )
        if st.button("Actualiser", key="refresh_cls_training_status"):
            st.rerun()

        if status.state in ACTIVE_STATES:
            st.info("Entraînement en cours")
            st.write(f"**Epoch :** {status.epoch_current} / {status.epochs_total}")
            st.write(f"**Progression :** {status.progress_percent:.0f} %")
            st.progress(min(1.0, max(0.0, status.progress_percent / 100.0)))
            st.write(f"- Modèle : `{status.model}`")
            st.write(f"- Résolution : {status.imgsz}")
            st.write(f"- Batch : {_batch_label(status.batch)}")
            st.write(f"- Device : {describe_device(status.device)} (`{status.device}`)")
            st.write(f"- Temps écoulé : {_format_duration(status.started_at, status.finished_at)}")
            log_tail = read_log_tail(run_dir, max_lines=50)
            st.markdown("#### Journal (dernières lignes)")
            show_training_log(log_tail)
        elif status.state == "completed":
            st.success("✓ Entraînement terminé")
            st.write(f"- Durée totale : {_format_duration(status.started_at, status.finished_at)}")
            export_name = (
                Path(status.export_model_path).name
                if status.export_model_path
                else None
            )
            if export_name:
                st.write(f"- Modèle export : `{export_name}`")
                st.caption(f"Chemin : `{status.export_model_path}`")
            st.write(f"- best.pt (Ultralytics) : `{status.best_model_path or 'non trouvé'}`")
            st.write(f"- last.pt : `{status.last_model_path or 'non trouvé'}`")
            st.write(f"- Accuracy Top-1 : {_format_metric(status.metrics.accuracy_top1)}")
            st.write(f"- Accuracy Top-5 : {_format_metric(status.metrics.accuracy_top5)}")
            st.write(f"- Dossier des résultats : `{run_dir}`")
        elif status.state == "failed":
            st.error("Échec de l'entraînement")
            st.write(status.error_message or "Erreur inconnue.")
            log_tail = read_log_tail(run_dir, max_lines=50)
            if log_tail:
                st.markdown("#### Journal")
                show_training_log(log_tail)
        elif status.state == "interrupted":
            st.warning("Entraînement interrompu")
            st.write(status.error_message or "Le processus d'entraînement n'existe plus.")


    @st.fragment(run_every=TRAINING_UI_REFRESH_SECONDS)
    def _live_run_followup(run_dir: Path) -> None:
        status = load_run_for_ui(run_dir)
        if status is None:
            st.warning("Impossible de lire le statut du run.")
            return
        _render_run_followup(status, run_dir)
        if not should_auto_refresh(status.state):
            st.rerun()


    active_run_dir: Path | None = None
    hint = st.session_state.get(SESSION_ACTIVE_CLS_RUN_KEY)
    if hint and Path(hint).is_dir():
        active_run_dir = Path(hint)
    disk_reserved = find_reserved_run(ARTIFACTS_RUNS_DIR)
    if disk_reserved is not None:
        active_run_dir = disk_reserved
        st.session_state[SESSION_ACTIVE_CLS_RUN_KEY] = str(disk_reserved)

    active_status = load_run_for_ui(active_run_dir) if active_run_dir and active_run_dir.is_dir() else None
    if active_status is None and hint and Path(hint).is_dir():
        active_run_dir = Path(hint)
        active_status = load_run_for_ui(active_run_dir)

    # Only treat classify runs as "our" follow-up; still lock if any run is active.
    any_training_locked = active_status is not None and active_status.state in ACTIVE_STATES
    active_is_classify = False
    if active_status is not None and active_run_dir is not None:
        active_is_classify = _run_task(active_run_dir, active_status) == "classify"

    training_locked = any_training_locked

    payload = st.session_state.get(SESSION_CLS_DATASET_KEY)
    dataset = None
    if payload:
        try:
            from vision_trainer.datasets.external import (
                SOURCE_EXTERNAL,
                assert_external_dataset_accessible,
                normalize_source_type,
                source_type_label_fr,
            )

            if normalize_source_type(payload.get("source_type")) == SOURCE_EXTERNAL:
                try:
                    assert_external_dataset_accessible(Path(payload["root"]))
                except Exception as exc:
                    st.error(str(exc))
                    payload = None
            if payload:
                dataset = classify_dataset_from_session_payload(payload)
        except (KeyError, TypeError, ValueError) as exc:
            st.error(
                f"Dataset classification en session invalide : {exc}. "
                "Réimportez-le depuis **Datasets** (mode Classification)."
            )
            dataset = None

    if dataset is None:
        if not training_locked:
            st.warning(
                "Aucun dataset de classification validé. "
                "Importez-en un sur **Datasets** (mode Classification)."
            )
    else:
        from vision_trainer.datasets.external import source_type_label_fr

        st.subheader("Dataset")
        name = payload.get("display_name") or payload.get("dataset_id")
        if name:
            st.write(
                f"**{name}**  ·  `{source_type_label_fr(payload.get('source_type'))}`"
            )
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Classes", dataset.num_classes)
        col_b.metric("Images train", dataset.train_image_count)
        col_c.metric("Images val", dataset.val_image_count)
        col_d.write("**Chemin**")
        col_d.code(str(dataset.root), language=None)
        if payload.get("dataset_id"):
            st.caption(f"Dataset ID : `{payload['dataset_id']}`")

    st.subheader("Configuration")

    model_key = st.selectbox(
        "Modèle",
        options=list(AVAILABLE_CLS_MODELS.keys()),
        index=list(AVAILABLE_CLS_MODELS.keys()).index(DEFAULT_CLS_MODEL_KEY),
        help="Poids préentraînés Ultralytics YOLO11 classification.",
        disabled=training_locked or dataset is None,
    )
    st.caption(f"Poids : `{AVAILABLE_CLS_MODELS[model_key]}`")

    epochs = st.number_input(
        "Epochs",
        min_value=1,
        value=DEFAULT_EPOCHS,
        step=1,
        disabled=training_locked or dataset is None,
    )
    imgsz = st.selectbox(
        "Taille d'image (imgsz)",
        options=list(CLS_IMGSZ_CHOICES),
        index=CLS_IMGSZ_CHOICES.index(DEFAULT_CLS_IMGSZ),
        disabled=training_locked or dataset is None,
    )
    batch_mode = st.selectbox(
        "Batch",
        options=["Auto (Ultralytics)", "8", "16", "32"],
        index=0,
        disabled=training_locked or dataset is None,
    )
    batch = BATCH_AUTO if str(batch_mode).startswith("Auto") else int(batch_mode)

    device_choice, resolved_device, resolved_label = render_device_selector(
        key_prefix="cls_train",
        disabled=training_locked or dataset is None,
        default_choice="auto",
    )

    if any_training_locked and not active_is_classify:
        st.warning(
            "Un entraînement de **détection** (ou autre tâche) est déjà en cours. "
            "Attendez sa fin avant de lancer une classification."
        )

    start = st.button(
        "Lancer l'entraînement (classification)",
        type="primary",
        disabled=training_locked or dataset is None,
    )

    if start and not training_locked and dataset is not None:
        request = ClassifyTrainingRequest(
            dataset=dataset,
            model_key=model_key,
            epochs=int(epochs),
            imgsz=int(imgsz),
            batch=batch,
            device_choice=device_choice,
            runs_root=ARTIFACTS_RUNS_DIR,
            dataset_id=(payload or {}).get("dataset_id"),
            dataset_source_type=(payload or {}).get("source_type"),
            dataset_name=(payload or {}).get("display_name"),
        )
        try:
            with st.spinner("Préparation du run…"):
                prepared = prepare_classify_training_run(request)
                start_training_subprocess(prepared.run_dir)
            st.session_state[SESSION_ACTIVE_CLS_RUN_KEY] = str(prepared.run_dir)
            st.success(f"Entraînement classification démarré (run `{prepared.run_id}`).")
            st.rerun()
        except TrainingError as exc:
            st.error(f"Impossible de démarrer l'entraînement : {exc}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Erreur inattendue au démarrage : {exc}")

    if active_status is not None and active_run_dir is not None and active_is_classify:
        if should_auto_refresh(active_status.state):
            _live_run_followup(active_run_dir)
        else:
            _render_run_followup(active_status, active_run_dir)
