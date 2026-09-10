from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from vision_trainer.training.device import describe_device
from vision_trainer.training.runs import ARTIFACTS_RUNS_DIR
from vision_trainer.training.session_dataset import (
    SESSION_DATASET_KEY,
    dataset_from_session_payload,
)
from vision_trainer.training.status import (
    ACTIVE_STATES,
    TRAINING_UI_REFRESH_SECONDS,
    find_reserved_run,
    read_log_tail,
    should_auto_refresh,
)
from vision_trainer.training.trainer import (
    AVAILABLE_MODELS,
    BATCH_AUTO,
    DEFAULT_EPOCHS,
    DEFAULT_IMGSZ,
    DEFAULT_MODEL_KEY,
    IMGSZ_CHOICES,
    SESSION_ACTIVE_RUN_KEY,
    TrainingError,
    TrainingRequest,
    load_run_for_ui,
    prepare_training_run,
    start_training_subprocess,
)
from vision_trainer.ui.device_selector import render_device_selector


def render() -> None:
    st.subheader("Détection")
    st.markdown("Lancez un entraînement Ultralytics (detect) ou suivez un run déjà démarré.")


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


    def _render_run_followup(status, run_dir: Path) -> None:
        """Render status, progress, metrics and log tail for one run (read-only UI)."""
        st.subheader("Suivi de l'entraînement")
        st.caption(f"Run : `{status.run_id}` — dossier `{run_dir}`")

        if st.button("Actualiser", key="refresh_training_status"):
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
            st.code(log_tail or "(journal encore vide)", language="text")

        elif status.state == "completed":
            st.success("✓ Entraînement terminé")
            st.write(f"- Durée totale : {_format_duration(status.started_at, status.finished_at)}")
            st.write(f"- best.pt : `{status.best_model_path or 'non trouvé'}`")
            st.write(f"- last.pt : `{status.last_model_path or 'non trouvé'}`")
            st.write(f"- Précision : {_format_metric(status.metrics.precision)}")
            st.write(f"- Rappel : {_format_metric(status.metrics.recall)}")
            st.write(f"- mAP50 : {_format_metric(status.metrics.map50)}")
            st.write(f"- mAP50-95 : {_format_metric(status.metrics.map50_95)}")
            st.write(f"- Dossier des résultats : `{run_dir}`")

        elif status.state == "failed":
            st.error("Échec de l'entraînement")
            st.write(status.error_message or "Erreur inconnue.")
            log_tail = read_log_tail(run_dir, max_lines=50)
            if log_tail:
                st.markdown("#### Journal")
                st.code(log_tail, language="text")

        elif status.state == "interrupted":
            st.warning("Entraînement interrompu")
            st.write(
                status.error_message
                or "Le processus d'entraînement n'existe plus. Le run a été marqué interrupted."
            )


    @st.fragment(run_every=TRAINING_UI_REFRESH_SECONDS)
    def _live_run_followup(run_dir: Path) -> None:
        """Poll status.json / train.log while the run is created or running."""
        status = load_run_for_ui(run_dir)
        if status is None:
            st.warning("Impossible de lire le statut du run.")
            return
        _render_run_followup(status, run_dir)
        # Leave the auto-refresh fragment once the run is terminal.
        if not should_auto_refresh(status.state):
            st.rerun()


    # Restore active/reserved run from disk (session_state is only a hint).
    active_run_dir: Path | None = None
    hint = st.session_state.get(SESSION_ACTIVE_RUN_KEY)
    if hint and Path(hint).is_dir():
        active_run_dir = Path(hint)
    disk_reserved = find_reserved_run(ARTIFACTS_RUNS_DIR)
    if disk_reserved is not None:
        active_run_dir = disk_reserved
        st.session_state[SESSION_ACTIVE_RUN_KEY] = str(disk_reserved)

    active_status = load_run_for_ui(active_run_dir) if active_run_dir and active_run_dir.is_dir() else None

    # Also allow selecting a recently finished run from session hint for display
    if active_status is None and hint and Path(hint).is_dir():
        active_run_dir = Path(hint)
        active_status = load_run_for_ui(active_run_dir)

    training_locked = active_status is not None and active_status.state in ACTIVE_STATES

    payload = st.session_state.get(SESSION_DATASET_KEY)
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
                except Exception as exc:  # ExternalDatasetError
                    st.error(str(exc))
                    payload = None
            if payload:
                dataset = dataset_from_session_payload(payload)
        except (KeyError, TypeError, ValueError) as exc:
            st.error(f"Dataset en session invalide : {exc}. Réimportez-le depuis la page Dataset.")
            dataset = None

    if dataset is None:
        if not training_locked:
            st.warning(
                "Aucun dataset valide n'est disponible pour lancer un **nouvel** entraînement. "
                "Importez-en un sur la rubrique **Datasets** (mode Détection). "
                "Le suivi d'un run existant reste accessible ci-dessous s'il y en a un."
            )
    else:
        if "train" not in dataset.splits:
            st.error("Le dataset validé ne contient pas de split train.")
            dataset = None
        else:
            from vision_trainer.datasets.external import source_type_label_fr

            train_count = dataset.splits["train"].image_count
            val_count = dataset.splits["val"].image_count if "val" in dataset.splits else 0
            st.subheader("Dataset")
            name = (payload or {}).get("display_name") or (payload or {}).get("dataset_id")
            if name:
                st.write(
                    f"**{name}**  ·  `{source_type_label_fr((payload or {}).get('source_type'))}`"
                )
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Classes", dataset.num_classes)
            col_b.metric("Images train", train_count)
            col_c.metric("Images val", val_count)
            col_d.write("**Chemin**")
            col_d.code(str(dataset.root), language=None)
            st.caption(f"data.yaml source : `{dataset.yaml_path}`")
            if payload.get("dataset_id"):
                st.caption(f"Dataset ID : `{payload['dataset_id']}`")

    st.subheader("Configuration")

    model_key = st.selectbox(
        "Modèle",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL_KEY),
        help="Poids préentraînés Ultralytics YOLO11.",
        disabled=training_locked or dataset is None,
    )
    st.caption(f"Poids : `{AVAILABLE_MODELS[model_key]}`")

    epochs = st.number_input(
        "Epochs",
        min_value=1,
        value=DEFAULT_EPOCHS,
        step=1,
        disabled=training_locked or dataset is None,
    )
    imgsz = st.selectbox(
        "Taille d'image (imgsz)",
        options=list(IMGSZ_CHOICES),
        index=IMGSZ_CHOICES.index(DEFAULT_IMGSZ),
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
        key_prefix="train",
        disabled=training_locked or dataset is None,
        default_choice="auto",
    )

    start = st.button(
        "Lancer l'entraînement",
        type="primary",
        disabled=training_locked or dataset is None,
    )

    if start and not training_locked and dataset is not None:
        request = TrainingRequest(
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
                prepared = prepare_training_run(request)
                start_training_subprocess(prepared.run_dir)
            st.session_state[SESSION_ACTIVE_RUN_KEY] = str(prepared.run_dir)
            st.success(f"Entraînement démarré (run `{prepared.run_id}`).")
            st.rerun()
        except TrainingError as exc:
            st.error(f"Impossible de démarrer l'entraînement : {exc}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Erreur inattendue au démarrage : {exc}")

    if active_status is not None and active_run_dir is not None:
        if should_auto_refresh(active_status.state):
            _live_run_followup(active_run_dir)
        else:
            _render_run_followup(active_status, active_run_dir)
