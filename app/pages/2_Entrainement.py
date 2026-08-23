from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from vision_trainer.training.device import describe_device, is_cuda_available, resolve_device
from vision_trainer.training.runs import ARTIFACTS_RUNS_DIR
from vision_trainer.training.session_dataset import (
    SESSION_DATASET_KEY,
    dataset_from_session_payload,
)
from vision_trainer.training.status import read_log_tail
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

st.set_page_config(page_title="Entraînement — Vision Trainer", layout="wide")
st.title("Entraînement YOLO")
st.markdown("Lancez un entraînement Ultralytics sur le dataset actuellement validé.")

payload = st.session_state.get(SESSION_DATASET_KEY)
if not payload:
    st.warning(
        "Aucun dataset valide n'est disponible. "
        "Importez et validez un dataset sur la page **Dataset** avant de lancer un entraînement."
    )
    st.stop()

try:
    dataset = dataset_from_session_payload(payload)
except (KeyError, TypeError, ValueError) as exc:
    st.error(f"Dataset en session invalide : {exc}. Réimportez-le depuis la page Dataset.")
    st.stop()

if "train" not in dataset.splits:
    st.error("Le dataset validé ne contient pas de split train. Impossible d'entraîner.")
    st.stop()

train_count = dataset.splits["train"].image_count
val_count = dataset.splits["val"].image_count if "val" in dataset.splits else 0

st.subheader("Dataset")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Classes", dataset.num_classes)
col_b.metric("Images train", train_count)
col_c.metric("Images val", val_count)
col_d.write("**Chemin**")
col_d.code(str(dataset.root), language=None)
st.caption(f"data.yaml source : `{dataset.yaml_path}`")


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


active_run_dir_raw = st.session_state.get(SESSION_ACTIVE_RUN_KEY)
active_run_dir = Path(active_run_dir_raw) if active_run_dir_raw else None
active_status = load_run_for_ui(active_run_dir) if active_run_dir and active_run_dir.is_dir() else None

if active_status is not None and active_run_dir is not None:
    st.subheader("Suivi de l'entraînement")
    st.caption(f"Run : `{active_status.run_id}` — dossier `{active_run_dir}`")

    if st.button("Actualiser", key="refresh_training_status"):
        st.rerun()

    if active_status.state in {"created", "running"}:
        st.info("Entraînement en cours")
        st.write(
            f"**Epoch :** {active_status.epoch_current} / {active_status.epochs_total}"
        )
        st.write(f"**Progression :** {active_status.progress_percent:.0f} %")
        st.progress(min(1.0, max(0.0, active_status.progress_percent / 100.0)))
        st.write(f"- Modèle : `{active_status.model}`")
        st.write(f"- Résolution : {active_status.imgsz}")
        st.write(f"- Batch : {_batch_label(active_status.batch)}")
        st.write(f"- Device : {describe_device(active_status.device)} (`{active_status.device}`)")
        st.write(
            f"- Temps écoulé : {_format_duration(active_status.started_at, active_status.finished_at)}"
        )

        log_tail = read_log_tail(active_run_dir, max_lines=50)
        st.markdown("#### Journal (dernières lignes)")
        st.code(log_tail or "(journal encore vide)", language="text")

    elif active_status.state == "completed":
        st.success("✓ Entraînement terminé")
        st.write(
            f"- Durée totale : {_format_duration(active_status.started_at, active_status.finished_at)}"
        )
        st.write(f"- best.pt : `{active_status.best_model_path or 'non trouvé'}`")
        st.write(f"- last.pt : `{active_status.last_model_path or 'non trouvé'}`")
        st.write(f"- Précision : {_format_metric(active_status.metrics.precision)}")
        st.write(f"- Rappel : {_format_metric(active_status.metrics.recall)}")
        st.write(f"- mAP50 : {_format_metric(active_status.metrics.map50)}")
        st.write(f"- mAP50-95 : {_format_metric(active_status.metrics.map50_95)}")
        st.write(f"- Dossier des résultats : `{active_run_dir}`")

    elif active_status.state == "failed":
        st.error("Échec de l'entraînement")
        st.write(active_status.error_message or "Erreur inconnue.")
        log_tail = read_log_tail(active_run_dir, max_lines=50)
        if log_tail:
            st.markdown("#### Journal")
            st.code(log_tail, language="text")

    elif active_status.state == "interrupted":
        st.warning("Entraînement interrompu")
        st.write(
            active_status.error_message
            or "Le processus d'entraînement n'existe plus. Le run a été marqué interrupted."
        )

st.subheader("Configuration")

training_locked = active_status is not None and active_status.state in {"created", "running"}

model_key = st.selectbox(
    "Modèle",
    options=list(AVAILABLE_MODELS.keys()),
    index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL_KEY),
    help="Poids préentraînés Ultralytics YOLO11.",
    disabled=training_locked,
)
st.caption(f"Poids : `{AVAILABLE_MODELS[model_key]}`")

epochs = st.number_input(
    "Epochs",
    min_value=1,
    value=DEFAULT_EPOCHS,
    step=1,
    disabled=training_locked,
)
imgsz = st.selectbox(
    "Taille d'image (imgsz)",
    options=list(IMGSZ_CHOICES),
    index=IMGSZ_CHOICES.index(DEFAULT_IMGSZ),
    disabled=training_locked,
)

batch_mode = st.selectbox(
    "Batch",
    options=["Auto (Ultralytics)", "8", "16", "32"],
    index=0,
    disabled=training_locked,
)
batch = BATCH_AUTO if batch_mode.startswith("Auto") else int(batch_mode)

cuda_available = is_cuda_available()
device_options = ["Auto", "CPU"]
if cuda_available:
    device_options.append("CUDA")

device_label = st.selectbox("Device", options=device_options, index=0, disabled=training_locked)
device_choice_map = {"Auto": "auto", "CPU": "cpu", "CUDA": "cuda"}
device_choice = device_choice_map[device_label]

try:
    resolved_device = resolve_device(device_choice)
    resolved_label = describe_device(resolved_device)
except Exception as exc:  # noqa: BLE001
    st.error(str(exc))
    st.stop()

st.info(f"Device réellement sélectionné : **{resolved_label}** (`{resolved_device}`)")
if device_choice == "auto":
    st.caption(
        "Mode Auto : CUDA détecté." if cuda_available else "Mode Auto : aucun GPU CUDA, utilisation du CPU."
    )

start = st.button(
    "Lancer l'entraînement",
    type="primary",
    disabled=training_locked,
)

if start and not training_locked:
    request = TrainingRequest(
        dataset=dataset,
        model_key=model_key,
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=batch,
        device_choice=device_choice,
        runs_root=ARTIFACTS_RUNS_DIR,
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
