"""Streamlit page: run a saved detection→classification pipeline on one image."""

from __future__ import annotations

import streamlit as st
from PIL import Image

from vision_trainer.inference.render import (
    ANNOTATION_SCALE_OPTIONS,
    annotated_image_to_jpeg_bytes,
    build_download_filename,
)
from vision_trainer.inference.uploads import SUPPORTED_UPLOAD_SUFFIXES, display_upload_name
from vision_trainer.pipeline.engine import (
    PipelineEngineError,
    result_table_rows,
    run_pipeline,
)
from vision_trainer.pipeline.render import draw_pipeline_result
from vision_trainer.pipeline.store import (
    PipelineStoreError,
    list_pipelines,
    load_pipeline,
    mapping_summary,
    resolve_run_weights,
)
from vision_trainer.ui.device_selector import render_device_selector

st.set_page_config(page_title="Inférence Pipeline — Vision Trainer", layout="wide")
st.title("Inférence Pipeline")
st.markdown(
    """
    Mode **Détection → Classification** : le détecteur trouve les objets,
    puis chaque classe configurée est affinée via un classificateur sur le **crop**.
    """
)


@st.cache_resource(show_spinner=False)
def _load_yolo_model(weights_path: str):
    from ultralytics import YOLO

    return YOLO(weights_path)


def _model_factory(weights: str):
    return _load_yolo_model(str(weights))


pipelines = list_pipelines()
if not pipelines:
    st.warning(
        "Aucun pipeline enregistré. Créez-en un sur la page **Pipelines** "
        "(détecteur + associations optionnelles)."
    )
    st.stop()

labels = [f"{p.name} ({p.pipeline_id})" for p in pipelines]
choice = st.selectbox("Pipeline", options=labels)
config = load_pipeline(pipelines[labels.index(choice)].pipeline_id)

st.subheader("Configuration")
st.markdown(f"- **Détecteur** : `{config.detector_run_id}`")
try:
    st.caption(f"Poids : `{resolve_run_weights(config.detector_run_id)}`")
except PipelineStoreError as exc:
    st.error(str(exc))
    st.stop()

active = [row for row in mapping_summary(config) if row["enabled"]]
if active:
    st.markdown("**Associations actives**")
    for row in active:
        st.markdown(
            f"- `{row['class_name']}` → `{row['classifier_run_id']}` "
            f"(conf ≥ {row['confidence_threshold']:.2f}, "
            f"écart ≥ {row['margin_threshold']:.2f}, top-{row['top_n']})"
        )
else:
    st.info("Aucune classe affinée : le pipeline se comporte comme une détection simple.")

device_choice = render_device_selector(key_prefix="pipeline_infer")
scale_labels = [label for label, _ in ANNOTATION_SCALE_OPTIONS]
scale_keys = {label: key for label, key in ANNOTATION_SCALE_OPTIONS}
scale_label = st.selectbox("Taille des annotations", options=scale_labels, index=0)
show_crops = st.checkbox("Afficher les crops analysés", value=False)

uploaded = st.file_uploader(
    "Image",
    type=[s.lstrip(".") for s in SUPPORTED_UPLOAD_SUFFIXES],
)
if uploaded is None:
    st.stop()

image = Image.open(uploaded).convert("RGB")
st.caption(f"Fichier : {display_upload_name(uploaded.name)} — {image.width}×{image.height} px")

if st.button("Lancer le pipeline", type="primary"):
    with st.spinner("Inférence pipeline…"):
        try:
            model_cache: dict = {}
            result = run_pipeline(
                image,
                config,
                device_choice=device_choice,
                model_cache=model_cache,
                model_factory=_model_factory,
            )
        except PipelineEngineError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Erreur pipeline : {exc}")
            st.stop()

    for warning in result.warnings:
        st.warning(warning)

    annotated = draw_pipeline_result(
        image,
        result,
        scale=scale_keys[scale_label],
    )
    st.image(annotated, caption="Résultat enrichi", use_container_width=True)
    st.download_button(
        "Télécharger l'image annotée",
        data=annotated_image_to_jpeg_bytes(annotated),
        file_name=build_download_filename(uploaded.name),
        mime="image/jpeg",
    )

    st.subheader("Tableau des résultats")
    rows = result_table_rows(result)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Aucune détection.")

    if show_crops:
        st.subheader("Crops analysés")
        refined_items = [
            item
            for item in result.items
            if item.refined and item.crop_box is not None
        ]
        if not refined_items:
            st.caption("Aucun crop (aucune classe affinée ou crop indisponible).")
        for index, item in enumerate(refined_items):
            left, top, right, bottom = item.crop_box  # type: ignore[misc]
            crop = image.crop((left, top, right, bottom))
            cols = st.columns([1, 2])
            with cols[0]:
                st.image(crop, caption=f"Crop #{index + 1}", use_container_width=True)
            with cols[1]:
                det = item.detection
                st.markdown(
                    f"**Détecté** : {det.class_name} ({det.confidence:.2%})  \n"
                    f"**Boîte crop** : `{item.crop_box}`"
                )
                if item.classification is None:
                    continue
                ref = item.classification
                st.markdown(f"**Statut** : `{ref.status}`")
                if ref.warning:
                    st.warning(ref.warning)
                if ref.class_name:
                    st.markdown(
                        f"**Classe** : {ref.class_name} "
                        f"({(ref.confidence or 0):.2%})"
                    )
                if ref.reason:
                    st.caption(ref.reason)
                if ref.top_n:
                    st.markdown(
                        "Top-N : "
                        + ", ".join(
                            f"{s.class_name} {s.confidence:.2f}" for s in ref.top_n
                        )
                    )
