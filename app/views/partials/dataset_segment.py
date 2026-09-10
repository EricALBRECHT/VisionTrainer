"""Streamlit: import and validate a YOLO segmentation dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from views.partials.dataset_source_ui import (
    render_external_dataset_picker,
    render_source_mode_selector,
    show_session_dataset_summary,
)
from vision_trainer.datasets.external import (
    SOURCE_EXTERNAL,
    SOURCE_UPLOADED,
    register_external_dataset_meta,
)
from vision_trainer.datasets.store import import_zip_to_persistent_dataset
from vision_trainer.segment.preview import SegmentPreviewError, draw_segment_preview
from vision_trainer.segment.session import (
    SESSION_SEG_DATASET_KEY,
    segment_dataset_from_session_payload,
    segment_dataset_to_session_payload,
    segment_info_from_session_payload,
)
from vision_trainer.segment.validator import validate_segment_dataset
from vision_trainer.yolo.parser import ZipExtractionError, list_images


def render() -> None:
    st.subheader("Segmentation")
    st.markdown(
        """
        **Détection** : où est l'objet (boîte)  
        **Classification** : de quoi s'agit-il  
        **Segmentation** : quels **pixels** appartiennent à l'objet (masque / contour)

        Source ZIP **ou** dossier monté (gros datasets).
        """
    )
    st.info(
        "Format YOLO segmentation : `train/images` + `train/labels` (+ `val/…`) et `data.yaml`. "
        "Les labels sont des **polygones** normalisés (`class x1 y1 x2 y2 …`), "
        "pas des bounding boxes de détection."
    )

    mode = render_source_mode_selector(key="seg_dataset_source_mode")
    if mode == "local":
        _render_external()
    else:
        _render_zip()


def _render_zip() -> None:
    uploaded_file = st.file_uploader(
        "Fichier ZIP du dataset de segmentation",
        type=["zip"],
        help="ZIP YOLO-seg (data.yaml + images/labels).",
        key="seg_zip_uploader",
    )

    if uploaded_file is None:
        if SESSION_SEG_DATASET_KEY in st.session_state:
            show_session_dataset_summary(
                st.session_state[SESSION_SEG_DATASET_KEY],
                title="Aucun nouvel upload. Le dernier dataset de segmentation validé reste disponible.",
            )
            return
        st.info("Importez un dataset ZIP, ou passez en **Dossier local**.")
        return

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
            return
        except OSError as exc:
            st.error(f"Erreur de lecture ou d'écriture lors de l'import : {exc}")
            return

        st.session_state.seg_dataset_upload_hash = upload_hash
        st.session_state.seg_dataset_extract_dir = str(extract_dir)
        st.session_state.seg_dataset_id = dataset_id

    if "seg_dataset_extract_dir" not in st.session_state:
        st.error("L'import du dataset a échoué. Réessayez avec une autre archive.")
        return

    extract_dir = Path(st.session_state.seg_dataset_extract_dir)
    result = validate_segment_dataset(extract_dir, containment_root=extract_dir)
    _show_segment_live(
        result,
        storage_path=extract_dir,
        source_type=SOURCE_UPLOADED,
        dataset_id=st.session_state.get("seg_dataset_id"),
        display_name=None,
        activate=True,
    )


def _render_external() -> None:
    path = render_external_dataset_picker(key_prefix="seg")
    if path is None:
        if SESSION_SEG_DATASET_KEY in st.session_state:
            show_session_dataset_summary(st.session_state[SESSION_SEG_DATASET_KEY])
        return

    pending_key = "seg_ext_pending"
    if st.button("Vérifier le dataset", type="primary", key="seg_ext_verify"):
        dataset_id, meta_dir = register_external_dataset_meta(
            path,
            task="segment",
            display_name=path.name,
        )
        result = validate_segment_dataset(
            path,
            containment_root=path,
            generated_yaml_dir=meta_dir,
        )
        segment_info = getattr(result, "segment_info", None)
        if result.dataset is None or not result.is_valid:
            st.session_state[pending_key] = {
                "path": str(path),
                "ok": False,
                "errors": [
                    i.message if not i.context else f"{i.message} ({i.context})"
                    for i in result.errors
                ],
                "warnings": [i.message for i in result.warnings],
                "infos": [i.message for i in result.infos],
            }
        else:
            payload = segment_dataset_to_session_payload(
                result.dataset,
                path,
                segment_info=segment_info,
                dataset_id=dataset_id,
                source_type=SOURCE_EXTERNAL,
                display_name=path.name,
            )
            st.session_state[pending_key] = {
                "path": str(path),
                "ok": True,
                "payload": payload,
                "errors": [],
                "warnings": [i.message for i in result.warnings],
                "infos": [i.message for i in result.infos],
            }

    pending = st.session_state.get(pending_key)
    if not pending or pending.get("path") != str(path):
        st.caption("Sélectionnez un dossier puis cliquez sur **Vérifier le dataset**.")
        return

    for msg in pending.get("errors") or []:
        st.error(msg)
    for msg in pending.get("warnings") or []:
        st.warning(msg)
    for msg in pending.get("infos") or []:
        st.info(msg)

    if not pending.get("ok"):
        st.error("Corrigez les erreurs avant d'entraîner.")
        return

    payload = pending["payload"]
    dataset = segment_dataset_from_session_payload(payload)
    segment_info = segment_info_from_session_payload(payload)
    train_n = dataset.splits["train"].image_count if "train" in dataset.splits else 0
    val_n = dataset.splits["val"].image_count if "val" in dataset.splits else 0
    instances = segment_info.total_instances if segment_info else 0
    st.success(
        f"✓ Dataset valide — Segmentation — train={train_n}, val={val_n}, "
        f"classes={dataset.num_classes}, instances={instances} — "
        "Source : externe — Aucun fichier copié"
    )
    if st.button("Utiliser ce dataset", type="primary", key="seg_use_external"):
        st.session_state[SESSION_SEG_DATASET_KEY] = payload
        st.success("Dataset externe sélectionné pour l'entraînement.")
        st.rerun()

    _preview_segment(dataset, key_suffix="external")


def _show_segment_live(
    result,
    *,
    storage_path: Path,
    source_type: str,
    dataset_id: str | None,
    display_name: str | None,
    activate: bool,
) -> None:
    dataset = result.dataset
    segment_info = getattr(result, "segment_info", None)
    if dataset is None:
        st.error("Impossible de charger le dataset de segmentation.")
        for issue in result.errors:
            st.error(issue.message if not issue.context else f"{issue.message} ({issue.context})")
        return

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
        if dataset_id:
            st.caption(f"Dataset ID : `{dataset_id}`")
        st.caption(f"Racine : `{dataset.root}`")

    with col_classes:
        st.subheader("Classes / instances")
        by_class = segment_info.instances_by_class() if segment_info else {}
        for class_id, class_name in sorted(dataset.class_names.items()):
            st.write(
                f"`{class_id}` — **{class_name}** : {by_class.get(class_name, 0)} instance(s)"
            )

    st.subheader("Validation")
    for issue in result.errors:
        st.error(issue.message if not issue.context else f"{issue.message} ({issue.context})")
    for issue in result.warnings:
        st.warning(issue.message)
    for issue in result.infos:
        st.info(issue.message)

    _preview_segment(dataset, key_suffix=source_type)

    if result.is_valid:
        payload = segment_dataset_to_session_payload(
            dataset,
            storage_path,
            segment_info=segment_info,
            dataset_id=dataset_id,
            source_type=source_type,
            display_name=display_name,
        )
        if activate:
            st.session_state[SESSION_SEG_DATASET_KEY] = payload
            st.success(
                "Dataset de segmentation validé. "
                "Passez à la rubrique **Entraînement** (mode Segmentation)."
            )
    else:
        st.error("Corrigez les erreurs avant d'entraîner.")


def _preview_segment(dataset, *, key_suffix: str) -> None:
    train_split = dataset.splits.get("train")
    val_split = dataset.splits.get("val")
    preview_split = train_split or val_split
    st.subheader("Prévisualisation des polygones")
    if (
        preview_split
        and preview_split.images_dir
        and preview_split.labels_dir
        and preview_split.images_dir.is_dir()
    ):
        images = list_images(preview_split.images_dir)[:12]
        if not images:
            st.caption("Aucune image à prévisualiser.")
            return
        labels = [path.name for path in images]
        choice = st.selectbox(
            "Image échantillon",
            options=labels,
            key=f"seg_preview_image_{key_suffix}",
        )
        image_path = images[labels.index(choice)]
        show_bbox = st.checkbox(
            "Afficher aussi la boîte englobante (lecture)",
            value=True,
            key=f"seg_preview_bbox_{key_suffix}",
        )
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
