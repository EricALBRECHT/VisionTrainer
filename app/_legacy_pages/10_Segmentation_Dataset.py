"""Streamlit: import and validate a YOLO segmentation dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from vision_trainer.datasets.store import import_zip_to_persistent_dataset
from vision_trainer.segment.preview import SegmentPreviewError, draw_segment_preview
from vision_trainer.segment.session import (
    SESSION_SEG_DATASET_KEY,
    segment_dataset_to_session_payload,
)
from vision_trainer.segment.validator import validate_segment_dataset
from vision_trainer.yolo.parser import ZipExtractionError, list_images

st.set_page_config(page_title="Dataset Segmentation — Vision Trainer", layout="wide")
st.title("Segmentation — Dataset")
st.markdown(
    """
    **Détection** : où est l'objet (boîte)  
    **Classification** : de quoi s'agit-il  
    **Segmentation** : quels **pixels** appartiennent à l'objet (masque / contour)
    """
)

st.info(
    "Format YOLO segmentation : `train/images` + `train/labels` (+ `val/…`) et `data.yaml`. "
    "Les labels sont des **polygones** normalisés (`class x1 y1 x2 y2 …`), "
    "pas des bounding boxes de détection."
)

uploaded_file = st.file_uploader(
    "Fichier ZIP du dataset de segmentation",
    type=["zip"],
    help="ZIP YOLO-seg (data.yaml + images/labels).",
)

if uploaded_file is None:
    if SESSION_SEG_DATASET_KEY in st.session_state:
        st.info("Aucun nouvel upload. Le dernier dataset de segmentation validé reste disponible.")
        payload = st.session_state[SESSION_SEG_DATASET_KEY]
        st.write(f"Dataset : `{payload.get('dataset_id', '—')}`")
        st.code(payload.get("root", ""), language=None)
        st.stop()
    st.info("Importez un dataset de segmentation au format ZIP pour commencer.")
    st.stop()

upload_bytes = uploaded_file.getvalue()
upload_hash = hashlib.sha256(upload_bytes).hexdigest()

if st.session_state.get("seg_dataset_upload_hash") != upload_hash:
    st.session_state.pop(SESSION_SEG_DATASET_KEY, None)
    st.session_state.pop("seg_dataset_extract_dir", None)
    st.session_state.pop("seg_dataset_id", None)

    try:
        dataset_id, extract_dir = import_zip_to_persistent_dataset(
            upload_bytes,
            content_hash=upload_hash,
            task="segment",
        )
    except ZipExtractionError as exc:
        st.error(f"Import refusé : {exc}")
        st.stop()
    except OSError as exc:
        st.error(f"Erreur de lecture ou d'écriture lors de l'import : {exc}")
        st.stop()

    st.session_state.seg_dataset_upload_hash = upload_hash
    st.session_state.seg_dataset_extract_dir = str(extract_dir)
    st.session_state.seg_dataset_id = dataset_id

if "seg_dataset_extract_dir" not in st.session_state:
    st.error("L'import du dataset a échoué. Réessayez avec une autre archive.")
    st.stop()

extract_dir = Path(st.session_state.seg_dataset_extract_dir)
result = validate_segment_dataset(extract_dir, containment_root=extract_dir)
dataset = result.dataset
segment_info = getattr(result, "segment_info", None)

if dataset is None:
    st.error("Impossible de charger le dataset de segmentation.")
    for issue in result.errors:
        st.error(issue.message if not issue.context else f"{issue.message} ({issue.context})")
    st.stop()

train_split = dataset.splits.get("train")
val_split = dataset.splits.get("val")
train_count = train_split.image_count if train_split else 0
val_count = val_split.image_count if val_split else 0

col_summary, col_classes = st.columns(2)
with col_summary:
    st.subheader("Résumé")
    st.metric("Nombre de classes", dataset.num_classes)
    st.write(f"- **train** : {train_count} image(s)")
    st.write(f"- **val** : {val_count} image(s)")
    if segment_info is not None:
        st.write(f"- **Instances** : {segment_info.total_instances}")
        means = [
            s.mean_points_per_polygon
            for s in segment_info.split_stats.values()
            if s.mean_points_per_polygon is not None
        ]
        if means:
            st.write(f"- **Points/polygone (moy.)** : {sum(means) / len(means):.1f}")
    st.caption(f"Dataset ID : `{st.session_state.get('seg_dataset_id', '—')}`")
    st.caption(f"Racine : `{dataset.root}`")

with col_classes:
    st.subheader("Classes / instances")
    by_class = segment_info.instances_by_class() if segment_info else {}
    for class_id, class_name in sorted(dataset.class_names.items()):
        st.write(f"`{class_id}` — **{class_name}** : {by_class.get(class_name, 0)} instance(s)")

st.subheader("Validation")
for issue in result.errors:
    st.error(issue.message if not issue.context else f"{issue.message} ({issue.context})")
for issue in result.warnings:
    st.warning(issue.message)
for issue in result.infos:
    st.info(issue.message)

st.subheader("Prévisualisation des polygones")
preview_split = train_split or val_split
if (
    preview_split
    and preview_split.images_dir
    and preview_split.labels_dir
    and preview_split.images_dir.is_dir()
):
    images = list_images(preview_split.images_dir)[:12]
    if not images:
        st.caption("Aucune image à prévisualiser.")
    else:
        labels = [path.name for path in images]
        choice = st.selectbox("Image échantillon", options=labels)
        image_path = images[labels.index(choice)]
        show_bbox = st.checkbox("Afficher aussi la boîte englobante (lecture)", value=True)
        try:
            preview = draw_segment_preview(
                image_path,
                preview_split.labels_dir,
                preview_split.images_dir,
                dataset.class_names,
                show_bbox=show_bbox,
            )
            st.image(preview, caption=image_path.name, use_container_width=True)
        except SegmentPreviewError as exc:
            st.warning(str(exc))
else:
    st.caption("Split images/labels indisponible pour la prévisualisation.")

if result.is_valid:
    st.session_state[SESSION_SEG_DATASET_KEY] = segment_dataset_to_session_payload(
        dataset,
        extract_dir,
        segment_info=segment_info,
        dataset_id=st.session_state.get("seg_dataset_id"),
    )
    st.success(
        "Dataset de segmentation validé. "
        "Passez à la page **Segmentation — Entraînement**."
    )
else:
    st.error("Corrigez les erreurs avant d'entraîner.")
