"""Streamlit page: create / edit detection→classify?→segment? pipelines."""

from __future__ import annotations

import streamlit as st

from vision_trainer.inference.discovery import discover_trained_models
from vision_trainer.pipeline.engine import load_detector_class_names
from vision_trainer.pipeline.models import (
    ClassificationStage,
    ClassMapping,
    PipelineConfig,
    SegmentationStage,
)
from vision_trainer.pipeline.store import (
    PipelineStoreError,
    delete_pipeline,
    duplicate_pipeline,
    list_pipelines,
    load_pipeline,
    make_pipeline_id,
    save_pipeline,
    validate_pipeline_config,
)
from vision_trainer.training.runs import ARTIFACTS_RUNS_DIR

st.set_page_config(page_title="Pipelines — Vision Trainer", layout="wide")
st.title("Pipelines")
st.markdown(
    """
    Chaînez un **détecteur** avec des raffinements **optionnels**, classe par classe :

    **Détection → Classification? → Segmentation?**

    Un classificateur / segmenter associé s'applique uniquement aux objets de cette classe,
    sur un **crop** de la boîte (image originale). Les deux étapes sont indépendantes.
    """
)


@st.cache_resource(show_spinner=False)
def _load_yolo_model(weights_path: str):
    from ultralytics import YOLO

    return YOLO(weights_path)


def _model_factory(weights: str):
    return _load_yolo_model(weights)


detect_models = discover_trained_models(ARTIFACTS_RUNS_DIR, task="detect")
classify_models = discover_trained_models(ARTIFACTS_RUNS_DIR, task="classify")
segment_models = discover_trained_models(ARTIFACTS_RUNS_DIR, task="segment")

if not detect_models:
    st.warning(
        "Aucun modèle de **détection** disponible (`weights/best.pt`). "
        "Entraînez un détecteur d'abord."
    )
    st.stop()

existing = list_pipelines()
mode = st.radio(
    "Action",
    options=["Créer", "Modifier", "Dupliquer", "Supprimer"],
    horizontal=True,
    key="pipeline_editor_mode",
)

pipeline: PipelineConfig | None = None
pipeline_id_locked = False

if mode == "Créer":
    default_name = "Mon pipeline"
    name = st.text_input("Nom du pipeline", value=default_name)
    detector_labels = [m.label for m in detect_models]
    selected_det_label = st.selectbox("Modèle de détection", options=detector_labels)
    selected_det = next(m for m in detect_models if m.label == selected_det_label)
    pipeline = PipelineConfig(
        pipeline_id=make_pipeline_id(name or "pipeline"),
        name=(name or "pipeline").strip(),
        detector_run_id=selected_det.run_id,
        mappings={},
    )
elif mode in {"Modifier", "Dupliquer", "Supprimer"}:
    if not existing:
        st.info("Aucun pipeline enregistré pour le moment.")
        st.stop()
    labels = [f"{p.name} ({p.pipeline_id})" for p in existing]
    choice = st.selectbox("Pipeline", options=labels)
    chosen = existing[labels.index(choice)]
    if mode == "Supprimer":
        st.warning(f"Supprimer définitivement « {chosen.name} » ?")
        if st.button("Confirmer la suppression", type="primary"):
            try:
                delete_pipeline(chosen.pipeline_id)
                st.success("Pipeline supprimé.")
                st.rerun()
            except PipelineStoreError as exc:
                st.error(str(exc))
        st.stop()
    if mode == "Dupliquer":
        new_name = st.text_input("Nouveau nom", value=f"{chosen.name} (copie)")
        if st.button("Dupliquer"):
            try:
                clone = duplicate_pipeline(
                    chosen.pipeline_id,
                    new_id=make_pipeline_id(new_name or chosen.name),
                    new_name=(new_name or chosen.name).strip(),
                )
                st.success(f"Copie créée : {clone.name} ({clone.pipeline_id})")
                st.rerun()
            except PipelineStoreError as exc:
                st.error(str(exc))
        st.stop()
    pipeline = load_pipeline(chosen.pipeline_id)
    pipeline_id_locked = True
    pipeline.name = st.text_input("Nom du pipeline", value=pipeline.name)
    if pipeline.detector_run_id not in {m.run_id for m in detect_models}:
        st.error(
            f"Détecteur « {pipeline.detector_run_id} » introuvable. "
            "Choisissez un autre modèle ou réentraînez."
        )
    detector_labels = [m.label for m in detect_models]
    default_det_index = 0
    for i, m in enumerate(detect_models):
        if m.run_id == pipeline.detector_run_id:
            default_det_index = i
            break
    selected_det_label = st.selectbox(
        "Modèle de détection",
        options=detector_labels,
        index=default_det_index,
    )
    selected_det = next(m for m in detect_models if m.label == selected_det_label)
    pipeline.detector_run_id = selected_det.run_id

assert pipeline is not None

st.subheader("Paramètres généraux")
col_a, col_b, col_c = st.columns(3)
with col_a:
    pipeline.crop_padding = st.slider(
        "Marge de crop (padding) défaut",
        min_value=0.0,
        max_value=0.30,
        value=float(pipeline.crop_padding),
        step=0.01,
        help="Utilisé pour classification, et pour segmentation si aucun padding dédié.",
    )
with col_b:
    pipeline.detect_conf = st.slider(
        "Confiance détection",
        min_value=0.05,
        max_value=0.95,
        value=float(pipeline.detect_conf),
        step=0.05,
    )
with col_c:
    pipeline.detect_iou = st.slider(
        "IoU détection",
        min_value=0.05,
        max_value=0.95,
        value=float(pipeline.detect_iou),
        step=0.05,
    )

st.subheader("Associations par classe")
st.caption(
    "Pour chaque classe détectée : classification et/ou segmentation optionnelles. "
    "Seuls les modèles `task=classify` / `task=segment` sont proposés."
)

try:
    class_names = load_detector_class_names(
        pipeline.detector_run_id,
        model_factory=_model_factory,
    )
except Exception as exc:  # noqa: BLE001
    st.error(f"Impossible de lire les classes du détecteur : {exc}")
    st.stop()

ordered_names = [class_names[i] for i in sorted(class_names)]
if not ordered_names:
    st.warning("Ce détecteur n'expose aucune classe.")
    st.stop()

classify_labels = [m.label for m in classify_models] if classify_models else []
segment_labels = [m.label for m in segment_models] if segment_models else []

updated_mappings: dict[str, ClassMapping] = {}
for class_name in ordered_names:
    previous = pipeline.mappings.get(class_name) or ClassMapping()
    with st.expander(f"Classe : **{class_name}**", expanded=previous.enabled):
        use_cls = st.checkbox(
            "Classification",
            value=previous.classification.enabled,
            key=f"pipe_cls_en_{pipeline.pipeline_id}_{class_name}",
        )
        use_seg = st.checkbox(
            "Segmentation",
            value=previous.segmentation.enabled,
            key=f"pipe_seg_en_{pipeline.pipeline_id}_{class_name}",
        )

        cls_stage = ClassificationStage(
            enabled=use_cls,
            run_id=previous.classification.run_id,
            confidence_threshold=previous.classification.confidence_threshold,
            margin_threshold=previous.classification.margin_threshold,
            top_n=previous.classification.top_n,
        )
        seg_stage = SegmentationStage(
            enabled=use_seg,
            run_id=previous.segmentation.run_id,
            confidence_threshold=previous.segmentation.confidence_threshold,
            crop_padding=previous.segmentation.crop_padding,
        )

        if use_cls:
            if not classify_models:
                st.warning("Aucun classificateur disponible.")
            else:
                default_cls = 0
                for i, m in enumerate(classify_models):
                    if m.run_id == cls_stage.run_id:
                        default_cls = i
                        break
                selected_cls_label = st.selectbox(
                    "Classificateur",
                    options=classify_labels,
                    index=default_cls,
                    key=f"pipe_cls_{pipeline.pipeline_id}_{class_name}",
                )
                selected_cls = next(
                    m for m in classify_models if m.label == selected_cls_label
                )
                cls_stage.run_id = selected_cls.run_id
                c1, c2, c3 = st.columns(3)
                with c1:
                    cls_stage.confidence_threshold = st.slider(
                        "Seuil confiance",
                        0.0,
                        1.0,
                        float(cls_stage.confidence_threshold),
                        0.01,
                        key=f"pipe_cf_{pipeline.pipeline_id}_{class_name}",
                    )
                with c2:
                    cls_stage.margin_threshold = st.slider(
                        "Écart Top1/Top2",
                        0.0,
                        1.0,
                        float(cls_stage.margin_threshold),
                        0.01,
                        key=f"pipe_mg_{pipeline.pipeline_id}_{class_name}",
                    )
                with c3:
                    cls_stage.top_n = int(
                        st.number_input(
                            "Top-N",
                            min_value=1,
                            max_value=20,
                            value=int(cls_stage.top_n),
                            key=f"pipe_tn_{pipeline.pipeline_id}_{class_name}",
                        )
                    )

        if use_seg:
            if not segment_models:
                st.warning("Aucun modèle de segmentation disponible.")
            else:
                default_seg = 0
                for i, m in enumerate(segment_models):
                    if m.run_id == seg_stage.run_id:
                        default_seg = i
                        break
                selected_seg_label = st.selectbox(
                    "Segmenter",
                    options=segment_labels,
                    index=default_seg,
                    key=f"pipe_seg_{pipeline.pipeline_id}_{class_name}",
                )
                selected_seg = next(
                    m for m in segment_models if m.label == selected_seg_label
                )
                seg_stage.run_id = selected_seg.run_id
                s1, s2 = st.columns(2)
                with s1:
                    seg_stage.confidence_threshold = st.slider(
                        "Seuil confiance segmentation",
                        0.05,
                        0.95,
                        float(seg_stage.confidence_threshold),
                        0.05,
                        key=f"pipe_scf_{pipeline.pipeline_id}_{class_name}",
                    )
                with s2:
                    pad_default = (
                        float(seg_stage.crop_padding)
                        if seg_stage.crop_padding is not None
                        else float(pipeline.crop_padding)
                    )
                    use_custom_pad = st.checkbox(
                        "Padding crop dédié",
                        value=seg_stage.crop_padding is not None,
                        key=f"pipe_spaden_{pipeline.pipeline_id}_{class_name}",
                    )
                    if use_custom_pad:
                        seg_stage.crop_padding = st.slider(
                            "Padding crop (segmentation)",
                            0.0,
                            0.30,
                            pad_default,
                            0.01,
                            key=f"pipe_spad_{pipeline.pipeline_id}_{class_name}",
                        )
                    else:
                        seg_stage.crop_padding = None

        updated_mappings[class_name] = ClassMapping(
            classification=cls_stage,
            segmentation=seg_stage,
        )

for class_name, mapping in pipeline.mappings.items():
    if class_name not in updated_mappings:
        updated_mappings[class_name] = ClassMapping(
            classification=ClassificationStage(
                enabled=False,
                run_id=mapping.classification.run_id,
                confidence_threshold=mapping.classification.confidence_threshold,
                margin_threshold=mapping.classification.margin_threshold,
                top_n=mapping.classification.top_n,
            ),
            segmentation=SegmentationStage(
                enabled=False,
                run_id=mapping.segmentation.run_id,
                confidence_threshold=mapping.segmentation.confidence_threshold,
                crop_padding=mapping.segmentation.crop_padding,
            ),
        )

pipeline.mappings = updated_mappings

st.divider()
if st.button("Enregistrer le pipeline", type="primary"):
    try:
        if mode == "Créer":
            pipeline.pipeline_id = make_pipeline_id(pipeline.name)
        validate_pipeline_config(pipeline, check_weights=True)
        path = save_pipeline(pipeline)
        st.success(f"Pipeline enregistré (format v2) : `{path.name}`")
        st.code(pipeline.pipeline_id)
    except PipelineStoreError as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Erreur : {exc}")

with st.expander("Aperçu JSON"):
    st.json(pipeline.to_dict())
