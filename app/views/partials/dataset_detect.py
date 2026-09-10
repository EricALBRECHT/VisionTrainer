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
from vision_trainer.training.session_dataset import (
    SESSION_DATASET_KEY,
    dataset_from_session_payload,
    dataset_to_session_payload,
)
from vision_trainer.yolo.parser import ZipExtractionError
from vision_trainer.yolo.validator import validate_dataset
from vision_trainer.yolo.visualization import ImagePreviewError, collect_sample_images, draw_annotations


def render() -> None:
    st.subheader("Détection")
    st.markdown(
        "Chargez un dataset YOLO (ZIP) **ou** sélectionnez un dossier monté "
        "sous la racine externe (gros datasets)."
    )

    mode = render_source_mode_selector(key="detect_dataset_source_mode")
    if mode == "local":
        _render_external()
    else:
        _render_zip()


def _render_zip() -> None:
    uploaded_file = st.file_uploader(
        "Fichier ZIP du dataset",
        type=["zip"],
        help=(
            "Le ZIP doit contenir un data.yaml, ou bien "
            "train/images + train/labels (+ val/…) avec un classes.txt."
        ),
        key="detect_zip_uploader",
    )

    if uploaded_file is None:
        if SESSION_DATASET_KEY in st.session_state:
            show_session_dataset_summary(
                st.session_state[SESSION_DATASET_KEY],
                title="Aucun nouvel upload. Le dernier dataset validé reste disponible.",
            )
            return
        st.info("Importez un dataset YOLO au format ZIP, ou passez en **Dossier local**.")
        return

    upload_bytes = uploaded_file.getvalue()
    upload_hash = hashlib.sha256(upload_bytes).hexdigest()

    if st.session_state.get("dataset_upload_hash") != upload_hash:
        st.session_state.pop(SESSION_DATASET_KEY, None)
        st.session_state.pop("dataset_extract_dir", None)
        st.session_state.pop("dataset_id", None)

        try:
            dataset_id, extract_dir = import_zip_to_persistent_dataset(
                upload_bytes,
                content_hash=upload_hash,
            )
        except ZipExtractionError as exc:
            st.error(f"Import refusé : {exc}")
            return
        except OSError as exc:
            st.error(f"Erreur de lecture ou d'écriture lors de l'import : {exc}")
            return

        st.session_state.dataset_upload_hash = upload_hash
        st.session_state.dataset_extract_dir = str(extract_dir)
        st.session_state.dataset_id = dataset_id

    if "dataset_extract_dir" not in st.session_state:
        st.error("L'import du dataset a échoué. Réessayez avec une autre archive.")
        return

    extract_dir = Path(st.session_state.dataset_extract_dir)
    result = validate_dataset(extract_dir, containment_root=extract_dir)
    _show_detect_validation_live(
        result,
        storage_path=extract_dir,
        source_type=SOURCE_UPLOADED,
        dataset_id=st.session_state.get("dataset_id"),
        display_name=None,
    )


def _render_external() -> None:
    path = render_external_dataset_picker(key_prefix="detect")
    if path is None:
        if SESSION_DATASET_KEY in st.session_state:
            show_session_dataset_summary(st.session_state[SESSION_DATASET_KEY])
        return

    pending_key = "detect_ext_pending"
    verify = st.button("Vérifier le dataset", type="primary", key="detect_ext_verify")
    if verify:
        dataset_id, meta_dir = register_external_dataset_meta(
            path,
            task="detect",
            display_name=path.name,
        )
        result = validate_dataset(
            path,
            containment_root=path,
            generated_yaml_dir=meta_dir,
        )
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
            payload = dataset_to_session_payload(
                result.dataset,
                path,
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
        st.error("Le dataset contient des erreurs bloquantes.")
        return

    payload = pending["payload"]
    dataset = dataset_from_session_payload(payload)
    train_n = dataset.splits["train"].image_count if "train" in dataset.splits else 0
    val_n = dataset.splits["val"].image_count if "val" in dataset.splits else 0
    st.success(
        f"✓ Dataset valide — Détection — train={train_n}, val={val_n}, "
        f"classes={dataset.num_classes} — Source : externe — Aucun fichier copié"
    )
    st.caption(f"Chemin : `{path}`")
    if st.button("Utiliser ce dataset", type="primary", key="detect_use_external"):
        st.session_state[SESSION_DATASET_KEY] = payload
        st.success("Dataset externe sélectionné pour l'entraînement.")
        st.rerun()

    _preview_detect(dataset, key_suffix="external")


def _show_detect_validation_live(
    result,
    *,
    storage_path: Path,
    source_type: str,
    dataset_id: str | None,
    display_name: str | None,
) -> None:
    dataset = result.dataset
    if dataset is None:
        st.error("Impossible de charger le dataset.")
        for issue in result.errors:
            st.error(issue.message if not issue.context else f"{issue.message} ({issue.context})")
        return

    col_summary, col_classes = st.columns(2)
    with col_summary:
        st.subheader("Résumé")
        st.metric("Nombre de classes", dataset.num_classes)
        available_splits = [name for name in ("train", "val", "test") if name in dataset.splits]
        st.write(
            "**Splits disponibles :**",
            ", ".join(available_splits) if available_splits else "Aucun",
        )
        if dataset_id:
            st.caption(f"Dataset ID : `{dataset_id}`")
        st.caption(f"Chemin : `{storage_path}`")
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
    for issue in result.errors:
        st.error(f"{issue.message}" + (f"\n\n`{issue.context}`" if issue.context else ""))
    for issue in result.warnings:
        st.warning(f"{issue.message}" + (f"\n\n`{issue.context}`" if issue.context else ""))
    for issue in result.infos:
        st.info(issue.message)

    if result.is_valid:
        st.success("Le dataset est valide.")
        st.session_state[SESSION_DATASET_KEY] = dataset_to_session_payload(
            dataset,
            storage_path,
            dataset_id=dataset_id,
            source_type=source_type,
            display_name=display_name,
        )
    else:
        st.error("Le dataset contient des erreurs bloquantes.")
        st.session_state.pop(SESSION_DATASET_KEY, None)

    _preview_detect(dataset, key_suffix=source_type)


def _preview_detect(dataset, *, key_suffix: str) -> None:
    available_splits = [name for name in ("train", "val", "test") if name in dataset.splits]
    preview_split = st.selectbox(
        "Split à prévisualiser",
        options=available_splits or ["train"],
        disabled=not available_splits,
        key=f"detect_preview_split_{key_suffix}",
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
