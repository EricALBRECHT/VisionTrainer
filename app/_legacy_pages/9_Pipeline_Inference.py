"""Streamlit page: run a multi-step pipeline on one image."""

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
from vision_trainer.pipeline.export import pipeline_result_json_bytes
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
    **Détection → Classification? → Segmentation?**  
    Le détecteur trouve les objets ; chaque classe configurée peut être classifiée
    et/ou segmentée sur le **crop** (image originale).
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
        "Aucun pipeline enregistré. Créez-en un sur la page **Pipelines**."
    )
    st.stop()

labels = [f"{p.name} ({p.pipeline_id})" for p in pipelines]
choice = st.selectbox("Pipeline", options=labels)
config = load_pipeline(pipelines[labels.index(choice)].pipeline_id)

st.subheader("Configuration")
st.markdown(
    f"- **Détecteur** : `{config.detector_run_id}`  \n"
    f"- **Format fichier** : v{config.format_version} (normalisé en mémoire)"
)
try:
    st.caption(f"Poids : `{resolve_run_weights(config.detector_run_id)}`")
except PipelineStoreError as exc:
    st.error(str(exc))
    st.stop()

active = [row for row in mapping_summary(config) if row["enabled"]]
if active:
    st.markdown("**Associations actives**")
    for row in active:
        bits = []
        if row["classification_enabled"]:
            bits.append(
                f"classify=`{row['classifier_run_id']}` "
                f"(conf≥{row['confidence_threshold']:.2f})"
            )
        if row["segmentation_enabled"]:
            bits.append(
                f"segment=`{row['segmenter_run_id']}` "
                f"(conf≥{row['segment_confidence_threshold']:.2f})"
            )
        st.markdown(f"- `{row['class_name']}` → " + " · ".join(bits))
else:
    st.info("Aucune classe affinée : détection simple.")

device_choice = render_device_selector(key_prefix="pipeline_infer")
scale_labels = [label for label, _ in ANNOTATION_SCALE_OPTIONS]
scale_keys = {label: key for label, key in ANNOTATION_SCALE_OPTIONS}
scale_label = st.selectbox("Taille des annotations", options=scale_labels, index=0)

st.subheader("Affichage")
c1, c2, c3, c4 = st.columns(4)
with c1:
    show_boxes = st.checkbox("Bounding boxes", value=True)
with c2:
    show_classification = st.checkbox("Classification", value=True)
with c3:
    show_masks = st.checkbox("Masques segmentation", value=True)
with c4:
    show_contours = st.checkbox("Contours", value=True)
show_seg_details = st.checkbox("Détails segmentation (labels masks)", value=False)
mask_opacity = st.slider("Opacité du masque", 0.05, 0.90, 0.40, 0.05)
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

    timings = result.timings
    st.caption(
        f"Temps — detect {timings.detection_ms:.0f} ms · "
        f"classify {timings.classification_ms:.0f} ms · "
        f"segment {timings.segmentation_ms:.0f} ms · "
        f"total {timings.total_ms:.0f} ms"
    )

    annotated = draw_pipeline_result(
        image,
        result,
        show_boxes=show_boxes,
        show_classification=show_classification,
        show_masks=show_masks,
        show_contours=show_contours,
        show_seg_labels=show_seg_details,
        mask_opacity=float(mask_opacity),
        scale=scale_keys[scale_label],
    )
    st.image(annotated, caption="Résultat enrichi", use_container_width=True)
    st.download_button(
        "Télécharger l'image annotée",
        data=annotated_image_to_jpeg_bytes(annotated),
        file_name=build_download_filename(uploaded.name),
        mime="image/jpeg",
    )
    st.download_button(
        "Exporter JSON",
        data=pipeline_result_json_bytes(result),
        file_name=f"pipeline_{config.pipeline_id}.json",
        mime="application/json",
    )

    st.subheader("Tableau des résultats")
    rows = result_table_rows(result)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption("« Surface crop » = part du **crop analysé** (pas une mesure physique).")
    else:
        st.info("Aucune détection.")

    with st.expander("Détails segmentation"):
        for index, item in enumerate(result.items):
            if not item.segmentations:
                continue
            st.markdown(
                f"**#{index + 1} {item.detection.class_name}** "
                f"({item.detection.confidence:.2%})"
            )
            for mask in item.segmentations:
                st.write(
                    f"- {mask.class_name} {mask.confidence:.2%} — "
                    f"{mask.mask_area_pixels} px — "
                    f"{mask.mask_area_ratio_crop * 100:.1f} % du crop — "
                    f"{mask.mask_area_ratio_image * 100:.2f} % de l'image"
                )

    if show_crops:
        st.subheader("Crops analysés")
        refined_items = [
            item for item in result.items if item.refined and item.crop_box is not None
        ]
        if not refined_items:
            st.caption("Aucun crop.")
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
                if item.classification is not None:
                    ref = item.classification
                    st.markdown(f"**Classification** : `{ref.status}`")
                    if ref.class_name:
                        st.markdown(
                            f"{ref.class_name} ({(ref.confidence or 0):.2%})"
                        )
                    if ref.warning:
                        st.warning(ref.warning)
                    if ref.top_n:
                        st.caption(
                            "Top-N : "
                            + ", ".join(
                                f"{s.class_name} {s.confidence:.2f}" for s in ref.top_n
                            )
                        )
                if item.segmentation is not None:
                    st.markdown(f"**Segmentation** : `{item.segmentation.status}`")
                    if item.segmentation.warning:
                        st.warning(item.segmentation.warning)
                    for mask in item.segmentations:
                        st.write(
                            f"- {mask.class_name} {mask.confidence:.2%} — "
                            f"{mask.mask_area_ratio_crop * 100:.1f} % du crop"
                        )
