from __future__ import annotations

import streamlit as st

from vision_trainer.training.device import describe_device, is_cuda_available, resolve_device
from vision_trainer.training.session_dataset import (
    SESSION_DATASET_KEY,
    dataset_from_session_payload,
)
from vision_trainer.training.trainer import (
    AVAILABLE_MODELS,
    BATCH_AUTO,
    DEFAULT_EPOCHS,
    DEFAULT_IMGSZ,
    DEFAULT_MODEL_KEY,
    IMGSZ_CHOICES,
    TrainingError,
    TrainingRequest,
    run_training,
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

st.subheader("Configuration")

model_key = st.selectbox(
    "Modèle",
    options=list(AVAILABLE_MODELS.keys()),
    index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL_KEY),
    help="Poids préentraînés Ultralytics YOLO11.",
)
st.caption(f"Poids : `{AVAILABLE_MODELS[model_key]}`")

epochs = st.number_input("Epochs", min_value=1, value=DEFAULT_EPOCHS, step=1)
imgsz = st.selectbox("Taille d'image (imgsz)", options=list(IMGSZ_CHOICES), index=IMGSZ_CHOICES.index(DEFAULT_IMGSZ))

batch_mode = st.selectbox(
    "Batch",
    options=["Auto (Ultralytics)", "8", "16", "32"],
    index=0,
)
batch = BATCH_AUTO if batch_mode.startswith("Auto") else int(batch_mode)

cuda_available = is_cuda_available()
device_options = ["Auto", "CPU"]
if cuda_available:
    device_options.append("CUDA")

device_label = st.selectbox("Device", options=device_options, index=0)
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

start = st.button("Lancer l'entraînement", type="primary")

if start:
    status = st.empty()
    status.info("Préparation…")

    request = TrainingRequest(
        dataset=dataset,
        model_key=model_key,
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=batch,
        device_choice=device_choice,
    )

    try:
        status.info("Entraînement en cours… Cela peut prendre plusieurs minutes.")
        result = run_training(request)
    except TrainingError as exc:
        status.error(f"Échec de l'entraînement : {exc}")
    except Exception as exc:  # noqa: BLE001
        status.error(f"Erreur inattendue : {exc}")
    else:
        status.success("Entraînement terminé.")
        st.write("**Résumé**")
        st.write(f"- Modèle : `{result.model_key}` (`{result.weights_name}`)")
        st.write(f"- Epochs : {result.epochs}")
        st.write(f"- imgsz : {result.imgsz}")
        st.write(f"- batch : {'auto' if result.batch == BATCH_AUTO else result.batch}")
        st.write(f"- device : {result.device_label} (`{result.device}`)")
        st.write(f"- Dossier des résultats : `{result.run_dir}`")
        st.write(f"- data.resolved.yaml : `{result.resolved_data_yaml}`")
        if result.best_weights is not None:
            st.success(f"Poids `best.pt` : `{result.best_weights}`")
        else:
            st.warning(
                "Le fichier `weights/best.pt` n'a pas été trouvé dans le dossier du run. "
                "Vérifiez les sorties Ultralytics dans le dossier des résultats."
            )
