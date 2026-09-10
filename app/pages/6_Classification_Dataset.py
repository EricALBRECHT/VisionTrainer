from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from vision_trainer.classify.session import (
    SESSION_CLS_DATASET_KEY,
    classify_dataset_to_session_payload,
)
from vision_trainer.classify.validator import class_size_guidance, validate_classify_dataset
from vision_trainer.datasets.store import import_zip_to_persistent_dataset
from vision_trainer.yolo.parser import ZipExtractionError

st.set_page_config(page_title="Dataset Classification — Vision Trainer", layout="wide")
st.title("Classification d'images — Dataset")
st.markdown(
    """
    Importez un dataset **par dossiers de classes** (comme Teachable Machine / ImageFolder).

    **Détection** (page Dataset YOLO) : image → objets + bounding boxes  
    **Classification** (cette page) : image → classe / probabilités
    """
)

st.info(
    "Structure attendue : `train/<Classe>/*.jpg` et idéalement `val/<Classe>/*.jpg`. "
    "Aucune bounding box n'est requise."
)

uploaded_file = st.file_uploader(
    "Fichier ZIP du dataset de classification",
    type=["zip"],
    help="ZIP contenant train/<classe>/… (val/ optionnel).",
)

if uploaded_file is None:
    if SESSION_CLS_DATASET_KEY in st.session_state:
        st.info("Aucun nouvel upload. Le dernier dataset de classification validé reste disponible.")
        payload = st.session_state[SESSION_CLS_DATASET_KEY]
        st.write(f"Dataset : `{payload.get('dataset_id', '—')}`")
        st.code(payload.get("root", ""), language=None)
        st.stop()
    st.info("Importez un dataset de classification au format ZIP pour commencer.")
    st.stop()

upload_bytes = uploaded_file.getvalue()
upload_hash = hashlib.sha256(upload_bytes).hexdigest()

if st.session_state.get("cls_dataset_upload_hash") != upload_hash:
    st.session_state.pop(SESSION_CLS_DATASET_KEY, None)
    st.session_state.pop("cls_dataset_extract_dir", None)
    st.session_state.pop("cls_dataset_id", None)

    try:
        dataset_id, extract_dir = import_zip_to_persistent_dataset(
            upload_bytes,
            content_hash=upload_hash,
            task="classify",
        )
    except ZipExtractionError as exc:
        st.error(f"Import refusé : {exc}")
        st.stop()
    except OSError as exc:
        st.error(f"Erreur de lecture ou d'écriture lors de l'import : {exc}")
        st.stop()

    st.session_state.cls_dataset_upload_hash = upload_hash
    st.session_state.cls_dataset_extract_dir = str(extract_dir)
    st.session_state.cls_dataset_id = dataset_id

if "cls_dataset_extract_dir" not in st.session_state:
    st.error("L'import du dataset a échoué. Réessayez avec une autre archive.")
    st.stop()

extract_dir = Path(st.session_state.cls_dataset_extract_dir)
result = validate_classify_dataset(extract_dir, containment_root=extract_dir)
dataset = result.dataset

if dataset is None:
    st.error("Impossible de charger le dataset de classification.")
    for issue in result.errors:
        st.error(issue.message if not issue.context else f"{issue.message} ({issue.context})")
    st.stop()

col_summary, col_classes = st.columns(2)
with col_summary:
    st.subheader("Résumé")
    st.metric("Nombre de classes", dataset.num_classes)
    st.metric("Images totales", dataset.total_images)
    st.write(f"- **train** : {dataset.train_image_count}")
    st.write(f"- **val** : {dataset.val_image_count}")
    st.write(f"- **test** : {dataset.test_image_count}")
    st.caption(f"Dataset ID : `{st.session_state.get('cls_dataset_id', '—')}`")
    st.caption(f"Racine : `{dataset.root}`")

with col_classes:
    st.subheader("Classes")
    for class_id, class_name in sorted(dataset.class_names.items()):
        stats = dataset.class_stats.get(class_name)
        total = stats.total if stats else 0
        st.write(
            f"`{class_id}` — **{class_name}** : {total} "
            f"(train={stats.train_count if stats else 0}, "
            f"val={stats.val_count if stats else 0}) — "
            f"*{class_size_guidance(total)}*"
        )

st.subheader("Validation")
for issue in result.errors:
    st.error(issue.message)
for issue in result.warnings:
    st.warning(issue.message)
for issue in result.infos:
    st.info(issue.message)

if result.is_valid:
    st.session_state[SESSION_CLS_DATASET_KEY] = classify_dataset_to_session_payload(
        dataset,
        extract_dir,
        dataset_id=st.session_state.get("cls_dataset_id"),
    )
    st.success(
        "Dataset de classification validé. "
        "Passez à la page **Classification — Entraînement**."
    )
else:
    st.error("Corrigez les erreurs avant d'entraîner.")
