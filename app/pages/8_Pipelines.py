"""Streamlit page: create / edit / save detection→classification pipelines."""

from __future__ import annotations

import streamlit as st

from vision_trainer.inference.discovery import discover_trained_models
from vision_trainer.pipeline.engine import load_detector_class_names
from vision_trainer.pipeline.models import ClassMapping, PipelineConfig
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
    Chaînez un **détecteur** avec des **classificateurs** optionnels, classe par classe.

    Exemple : détecter `Apple` / `Tomato`, puis affiner uniquement `Apple` vers
    Golden / Gala / Granny Smith. Les classes sans association restent des détections simples.

    Un classificateur associé sera appliqué **uniquement** aux objets détectés dans cette classe,
    sur un **crop** de la boîte (image originale), pas sur toute l'image.
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
    # Modifier
    pipeline = load_pipeline(chosen.pipeline_id)
    pipeline_id_locked = True
    pipeline.name = st.text_input("Nom du pipeline", value=pipeline.name)
    det_by_id = {m.run_id: m for m in detect_models}
    if pipeline.detector_run_id not in det_by_id:
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
        "Marge de crop (padding)",
        min_value=0.0,
        max_value=0.30,
        value=float(pipeline.crop_padding),
        step=0.01,
        help="Fraction de la largeur/hauteur de la boîte ajoutée de chaque côté.",
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
    "Cochez « Affiner » pour appliquer un classificateur au crop de chaque détection "
    "de cette classe. Les classificateurs proposés sont uniquement des modèles `task=classify`."
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

updated_mappings: dict[str, ClassMapping] = {}
for class_name in ordered_names:
    previous = pipeline.mappings.get(class_name) or ClassMapping()
    with st.expander(f"Classe : **{class_name}**", expanded=previous.enabled):
        enabled = st.checkbox(
            "Affiner cette classe",
            value=previous.enabled,
            key=f"pipe_en_{pipeline.pipeline_id}_{class_name}",
        )
        mapping = ClassMapping(
            enabled=enabled,
            classifier_run_id=previous.classifier_run_id,
            confidence_threshold=previous.confidence_threshold,
            margin_threshold=previous.margin_threshold,
            top_n=previous.top_n,
        )
        if enabled:
            if not classify_models:
                st.warning("Aucun classificateur disponible. Entraînez un modèle classify.")
            else:
                default_cls = 0
                for i, m in enumerate(classify_models):
                    if m.run_id == mapping.classifier_run_id:
                        default_cls = i
                        break
                selected_cls_label = st.selectbox(
                    "Classificateur",
                    options=classify_labels,
                    index=default_cls,
                    key=f"pipe_cls_{pipeline.pipeline_id}_{class_name}",
                )
                selected_cls = next(m for m in classify_models if m.label == selected_cls_label)
                mapping.classifier_run_id = selected_cls.run_id
                c1, c2, c3 = st.columns(3)
                with c1:
                    mapping.confidence_threshold = st.slider(
                        "Seuil confiance",
                        0.0,
                        1.0,
                        float(mapping.confidence_threshold),
                        0.01,
                        key=f"pipe_cf_{pipeline.pipeline_id}_{class_name}",
                    )
                with c2:
                    mapping.margin_threshold = st.slider(
                        "Écart Top1/Top2",
                        0.0,
                        1.0,
                        float(mapping.margin_threshold),
                        0.01,
                        key=f"pipe_mg_{pipeline.pipeline_id}_{class_name}",
                    )
                with c3:
                    mapping.top_n = st.number_input(
                        "Top-N",
                        min_value=1,
                        max_value=20,
                        value=int(mapping.top_n),
                        key=f"pipe_tn_{pipeline.pipeline_id}_{class_name}",
                    )
        updated_mappings[class_name] = mapping

# Keep orphan mappings (detector classes removed) disabled in file for transparency
for class_name, mapping in pipeline.mappings.items():
    if class_name not in updated_mappings:
        updated_mappings[class_name] = ClassMapping(
            enabled=False,
            classifier_run_id=mapping.classifier_run_id,
            confidence_threshold=mapping.confidence_threshold,
            margin_threshold=mapping.margin_threshold,
            top_n=mapping.top_n,
        )

pipeline.mappings = updated_mappings
if not pipeline_id_locked and mode == "Créer":
    # Refresh id from current name when creating
    pass

st.divider()
if st.button("Enregistrer le pipeline", type="primary"):
    try:
        if mode == "Créer":
            pipeline.pipeline_id = make_pipeline_id(pipeline.name)
        validate_pipeline_config(pipeline, check_weights=True)
        path = save_pipeline(pipeline)
        st.success(f"Pipeline enregistré : `{path.name}`")
        st.code(pipeline.pipeline_id)
    except PipelineStoreError as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Erreur : {exc}")

with st.expander("Aperçu JSON"):
    st.json(pipeline.to_dict())
