from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from views.partials.dataset_source_ui import (
    render_external_dataset_picker,
    render_source_mode_selector,
    show_session_dataset_summary,
)
from vision_trainer.classify.session import (
    SESSION_CLS_DATASET_KEY,
    classify_dataset_to_session_payload,
)
from vision_trainer.classify.validator import class_size_guidance, validate_classify_dataset
from vision_trainer.datasets.external import (
    SOURCE_EXTERNAL,
    SOURCE_UPLOADED,
    register_external_dataset_meta,
)
from vision_trainer.datasets.store import import_zip_to_persistent_dataset
from vision_trainer.yolo.parser import ZipExtractionError


def render() -> None:
    st.subheader("Classification")
    st.markdown(
        """
        Importez un dataset **par dossiers de classes** (ZIP) **ou** sélectionnez
        un dossier monté sous la racine externe.

        **Détection** : image → objets + bounding boxes  
        **Classification** : image → classe / probabilités
        """
    )
    st.info(
        "Structure attendue : `train/<Classe>/*.jpg` et idéalement `val/<Classe>/*.jpg`. "
        "Aucune bounding box n'est requise."
    )

    mode = render_source_mode_selector(key="cls_dataset_source_mode")
    if mode == "local":
        _render_external()
    else:
        _render_zip()


def _render_zip() -> None:
    uploaded_file = st.file_uploader(
        "Fichier ZIP du dataset de classification",
        type=["zip"],
        help="ZIP contenant train/<classe>/… (val/ optionnel).",
        key="cls_zip_uploader",
    )

    if uploaded_file is None:
        if SESSION_CLS_DATASET_KEY in st.session_state:
            show_session_dataset_summary(
                st.session_state[SESSION_CLS_DATASET_KEY],
                title="Aucun nouvel upload. Le dernier dataset de classification validé reste disponible.",
            )
            return
        st.info("Importez un dataset ZIP, ou passez en **Dossier local**.")
        return

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
            return
        except OSError as exc:
            st.error(f"Erreur de lecture ou d'écriture lors de l'import : {exc}")
            return

        st.session_state.cls_dataset_upload_hash = upload_hash
        st.session_state.cls_dataset_extract_dir = str(extract_dir)
        st.session_state.cls_dataset_id = dataset_id

    if "cls_dataset_extract_dir" not in st.session_state:
        st.error("L'import du dataset a échoué. Réessayez avec une autre archive.")
        return

    extract_dir = Path(st.session_state.cls_dataset_extract_dir)
    result = validate_classify_dataset(extract_dir, containment_root=extract_dir)
    _show_classify_result(
        result,
        storage_path=extract_dir,
        source_type=SOURCE_UPLOADED,
        dataset_id=st.session_state.get("cls_dataset_id"),
        display_name=None,
        activate=True,
    )


def _render_external() -> None:
    path = render_external_dataset_picker(key_prefix="cls")
    if path is None:
        if SESSION_CLS_DATASET_KEY in st.session_state:
            show_session_dataset_summary(st.session_state[SESSION_CLS_DATASET_KEY])
        return

    pending_key = "cls_ext_pending"
    if st.button("Vérifier le dataset", type="primary", key="cls_ext_verify"):
        dataset_id, _meta_dir = register_external_dataset_meta(
            path,
            task="classify",
            display_name=path.name,
        )
        result = validate_classify_dataset(path, containment_root=path)
        if result.dataset is None or not result.is_valid:
            st.session_state[pending_key] = {
                "path": str(path),
                "ok": False,
                "errors": [i.message for i in result.errors],
                "warnings": [i.message for i in result.warnings],
                "infos": [i.message for i in result.infos],
            }
        else:
            payload = classify_dataset_to_session_payload(
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
                "summary": {
                    "num_classes": result.dataset.num_classes,
                    "train": result.dataset.train_image_count,
                    "val": result.dataset.val_image_count,
                    "total": result.dataset.total_images,
                    "class_lines": [
                        (
                            cid,
                            name,
                            (result.dataset.class_stats[name].total
                             if name in result.dataset.class_stats else 0),
                            (
                                result.dataset.class_stats[name].train_count
                                if name in result.dataset.class_stats
                                else 0
                            ),
                            (
                                result.dataset.class_stats[name].val_count
                                if name in result.dataset.class_stats
                                else 0
                            ),
                        )
                        for cid, name in sorted(result.dataset.class_names.items())
                    ],
                },
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

    summary = pending["summary"]
    st.success(
        f"✓ Dataset valide — Classification — train={summary['train']}, "
        f"val={summary['val']}, classes={summary['num_classes']} — "
        "Source : externe — Aucun fichier copié"
    )
    for cid, name, total, tr, va in summary["class_lines"]:
        st.write(
            f"`{cid}` — **{name}** : {total} (train={tr}, val={va}) — "
            f"*{class_size_guidance(total)}*"
        )
    if st.button("Utiliser ce dataset", type="primary", key="cls_use_external"):
        st.session_state[SESSION_CLS_DATASET_KEY] = pending["payload"]
        st.success("Dataset externe sélectionné pour l'entraînement.")
        st.rerun()


def _show_classify_result(
    result,
    *,
    storage_path: Path,
    source_type: str,
    dataset_id: str | None,
    display_name: str | None,
    activate: bool,
) -> None:
    dataset = result.dataset
    if dataset is None:
        st.error("Impossible de charger le dataset de classification.")
        for issue in result.errors:
            st.error(issue.message if not issue.context else f"{issue.message} ({issue.context})")
        return

    col_summary, col_classes = st.columns(2)
    with col_summary:
        st.subheader("Résumé")
        st.metric("Nombre de classes", dataset.num_classes)
        st.metric("Images totales", dataset.total_images)
        st.write(f"- **train** : {dataset.train_image_count}")
        st.write(f"- **val** : {dataset.val_image_count}")
        st.write(f"- **test** : {dataset.test_image_count}")
        if dataset_id:
            st.caption(f"Dataset ID : `{dataset_id}`")
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
        payload = classify_dataset_to_session_payload(
            dataset,
            storage_path,
            dataset_id=dataset_id,
            source_type=source_type,
            display_name=display_name,
        )
        if activate:
            st.session_state[SESSION_CLS_DATASET_KEY] = payload
            st.success(
                "Dataset de classification validé. "
                "Passez à la rubrique **Entraînement** (mode Classification)."
            )
    else:
        st.error("Corrigez les erreurs avant d'entraîner.")
