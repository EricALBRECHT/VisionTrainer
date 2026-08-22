from __future__ import annotations

import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from vision_trainer.yolo.parser import ZipExtractionError, extract_zip_dataset
from vision_trainer.yolo.validator import validate_dataset
from vision_trainer.yolo.visualization import ImagePreviewError, collect_sample_images, draw_annotations
from vision_trainer.training.session_dataset import SESSION_DATASET_KEY, dataset_to_session_payload

st.set_page_config(page_title="Dataset — Vision Trainer", layout="wide")
st.title("Dataset YOLO")
st.markdown("Chargez un dataset YOLO (ZIP) pour l'analyser et le valider.")

uploaded_file = st.file_uploader(
    "Fichier ZIP du dataset",
    type=["zip"],
    help="Le ZIP doit contenir un fichier data.yaml à la racine ou dans un sous-dossier.",
)

if uploaded_file is None:
    st.info("Importez un dataset YOLO au format ZIP pour commencer.")
    st.stop()

upload_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
if st.session_state.get("dataset_upload_hash") != upload_hash:
    if st.session_state.get("dataset_temp_dir"):
        shutil.rmtree(st.session_state["dataset_temp_dir"], ignore_errors=True)
        st.session_state.pop("dataset_temp_dir", None)
        st.session_state.pop("dataset_extract_dir", None)
        st.session_state.pop("dataset_upload_hash", None)
        st.session_state.pop(SESSION_DATASET_KEY, None)

    temp_root = Path(tempfile.mkdtemp(prefix="vision-trainer-dataset-"))
    zip_path = temp_root / "dataset.zip"
    extract_dir = temp_root / "extracted"

    try:
        zip_path.write_bytes(uploaded_file.getvalue())
        extract_zip_dataset(zip_path, extract_dir)
    except ZipExtractionError as exc:
        shutil.rmtree(temp_root, ignore_errors=True)
        st.error(f"Import refusé pour des raisons de sécurité : {exc}")
        st.stop()
    except zipfile.BadZipFile:
        shutil.rmtree(temp_root, ignore_errors=True)
        st.error("Archive ZIP invalide ou corrompue.")
        st.stop()
    except OSError as exc:
        shutil.rmtree(temp_root, ignore_errors=True)
        st.error(f"Erreur de lecture ou d'écriture lors de l'import : {exc}")
        st.stop()

    st.session_state.dataset_upload_hash = upload_hash
    st.session_state.dataset_temp_dir = str(temp_root)
    st.session_state.dataset_extract_dir = str(extract_dir)

if "dataset_extract_dir" not in st.session_state:
    st.error("L'import du dataset a échoué. Réessayez avec une autre archive.")
    st.stop()

extract_dir = Path(st.session_state.dataset_extract_dir)
result = validate_dataset(extract_dir)
dataset = result.dataset

if dataset is None:
    st.error("Impossible de charger le dataset.")
    for issue in result.errors:
        st.error(issue.message if not issue.context else f"{issue.message} ({issue.context})")
    st.stop()

col_summary, col_classes = st.columns(2)

with col_summary:
    st.subheader("Résumé")
    st.metric("Nombre de classes", dataset.num_classes)
    available_splits = [name for name in ("train", "val", "test") if name in dataset.splits]
    st.write("**Splits disponibles :**", ", ".join(available_splits) if available_splits else "Aucun")

    for split_name in available_splits:
        split = dataset.splits[split_name]
        st.write(f"- **{split_name}** : {split.image_count} image(s)")

with col_classes:
    st.subheader("Classes")
    if dataset.class_names:
        for class_id, class_name in sorted(dataset.class_names.items()):
            st.write(f"`{class_id}` — {class_name}")
    else:
        st.write("Aucune classe définie.")

st.subheader("Validation")

errors = result.errors
warnings = result.warnings
infos = result.infos

if errors:
    st.markdown("#### Erreurs bloquantes")
    for issue in errors:
        if issue.context:
            st.error(f"{issue.message}\n\n`{issue.context}`")
        else:
            st.error(issue.message)

if warnings:
    st.markdown("#### Avertissements")
    for issue in warnings:
        if issue.context:
            st.warning(f"{issue.message}\n\n`{issue.context}`")
        else:
            st.warning(issue.message)

if infos:
    st.markdown("#### Informations")
    for issue in infos:
        st.info(issue.message)

if result.is_valid:
    st.success("Le dataset est valide.")
    st.session_state[SESSION_DATASET_KEY] = dataset_to_session_payload(dataset, extract_dir)
else:
    st.error("Le dataset contient des erreurs bloquantes.")
    st.session_state.pop(SESSION_DATASET_KEY, None)

preview_split = st.selectbox(
    "Split à prévisualiser",
    options=available_splits or ["train"],
    disabled=not available_splits,
)

if preview_split and preview_split in dataset.splits:
    split = dataset.splits[preview_split]
    if split.images_dir and split.images_dir.is_dir():
        st.subheader(f"Aperçu — split {preview_split}")
        samples = collect_sample_images(
            split.images_dir,
            split.labels_dir,
            dataset.class_names,
            limit=6,
        )
        if not samples:
            st.write("Aucune image à afficher.")
        else:
            columns = st.columns(min(3, len(samples)))
            for index, (image_path, annotations) in enumerate(samples):
                with columns[index % len(columns)]:
                    try:
                        annotated = draw_annotations(image_path, annotations)
                    except ImagePreviewError as exc:
                        st.warning(str(exc))
                        continue
                    st.image(
                        annotated,
                        caption=f"{image_path.name} ({len(annotations)} bbox)",
                        use_container_width=True,
                    )
