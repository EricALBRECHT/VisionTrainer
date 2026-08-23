from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from vision_trainer.inference.discovery import discover_trained_models
from vision_trainer.inference.predictor import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    InferenceError,
    load_image_rgb,
    run_inference,
)
from vision_trainer.inference.render import (
    annotated_image_to_jpeg_bytes,
    build_download_filename,
    draw_detections,
)
from vision_trainer.results.models import SESSION_INFERENCE_WEIGHTS_KEY
from vision_trainer.training.device import describe_device, is_cuda_available, resolve_device
from vision_trainer.training.runs import ARTIFACTS_RUNS_DIR

st.set_page_config(page_title="Inférence — Vision Trainer", layout="wide")
st.title("Inférence")
st.markdown("Testez un modèle entraîné sur une image.")

models = discover_trained_models(ARTIFACTS_RUNS_DIR)
if not models:
    st.warning(
        "Aucun modèle entraîné disponible. "
        "Lancez un entraînement jusqu'à obtenir `weights/best.pt` dans `artifacts/runs/`."
    )
    st.stop()

model_labels = [model.label for model in models]
preferred_weights = st.session_state.get(SESSION_INFERENCE_WEIGHTS_KEY)
default_index = 0
if preferred_weights:
    for index, model in enumerate(models):
        if Path(model.weights_path).resolve() == Path(preferred_weights).resolve():
            default_index = index
            break

selected_label = st.selectbox("Modèle", options=model_labels, index=default_index)
selected_model = next(model for model in models if model.label == selected_label)
st.caption(f"Poids : `{selected_model.weights_path}`")


@st.cache_resource(show_spinner=False)
def _load_yolo_model(weights_path: str):
    from ultralytics import YOLO

    return YOLO(weights_path)


uploaded = st.file_uploader(
    "Image à analyser",
    type=["jpg", "jpeg", "png", "webp"],
    help="Formats acceptés : JPG, JPEG, PNG, WEBP.",
)

original_image: Image.Image | None = None
original_name = "image.jpg"

if uploaded is not None:
    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        st.error("Format d'image non supporté.")
        st.stop()

    temp_dir = Path(st.session_state.get("inference_temp_dir") or "")
    if not temp_dir or not temp_dir.exists():
        import tempfile

        temp_dir = Path(tempfile.mkdtemp(prefix="vision-trainer-inference-"))
        st.session_state["inference_temp_dir"] = str(temp_dir)

    image_path = temp_dir / uploaded.name
    image_path.write_bytes(uploaded.getvalue())
    original_name = uploaded.name

    try:
        original_image = load_image_rgb(image_path)
    except InferenceError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Image originale")
    st.image(original_image, use_container_width=True)

st.subheader("Paramètres")
conf = st.slider("Confidence threshold", min_value=0.05, max_value=0.95, value=DEFAULT_CONF, step=0.05)
iou = st.slider("IoU threshold", min_value=0.05, max_value=0.95, value=DEFAULT_IOU, step=0.05)

cuda_available = is_cuda_available()
device_options = ["Auto", "CPU"]
if cuda_available:
    device_options.append("CUDA")
device_label = st.selectbox("Device", options=device_options, index=0)
device_choice_map = {"Auto": "auto", "CPU": "cpu", "CUDA": "cuda"}
device_choice = device_choice_map[device_label]

try:
    resolved_device = resolve_device(device_choice)
except Exception as exc:  # noqa: BLE001
    st.error(str(exc))
    st.stop()

st.info(f"Device réellement sélectionné : **{describe_device(resolved_device)}** (`{resolved_device}`)")

can_run = original_image is not None and bool(selected_model.weights_path)
run_clicked = st.button("Lancer l'inférence", type="primary", disabled=not can_run)

if run_clicked:
    if original_image is None:
        st.error("Chargez une image avant de lancer l'inférence.")
        st.stop()

    try:
        with st.spinner("Inférence en cours…"):
            # Cache is keyed by weights path; factory uses the cached model instance.
            cached_model = _load_yolo_model(selected_model.weights_path)

            def _factory(_path: str):
                return cached_model

            result = run_inference(
                weights_path=selected_model.weights_path,
                image=original_image,
                conf=float(conf),
                iou=float(iou),
                device_choice=device_choice,
                model_factory=_factory,
            )
            annotated = draw_detections(original_image, result.detections)
    except InferenceError as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Erreur inattendue pendant l'inférence : {exc}")
    else:
        st.session_state["inference_last_result"] = {
            "detections": [
                {
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "x1": det.x1,
                    "y1": det.y1,
                    "x2": det.x2,
                    "y2": det.y2,
                }
                for det in result.detections
            ],
            "annotated_jpeg": annotated_image_to_jpeg_bytes(annotated),
            "original_name": original_name,
        }

# Display last successful inference for the current session (survives widget reruns)
last = st.session_state.get("inference_last_result")
if last:
    st.subheader("Résultat")
    detections = last["detections"]
    annotated_bytes = last["annotated_jpeg"]

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Originale**")
        if original_image is not None:
            st.image(original_image, use_container_width=True)
        else:
            st.caption("Rechargez une image pour revoir l'originale à côté du résultat.")
    with col_right:
        st.markdown("**Annotée**")
        st.image(annotated_bytes, use_container_width=True)

    st.write(f"**Nombre total de détections :** {len(detections)}")

    if not detections:
        st.info("Aucune détection au-dessus du seuil sélectionné.")
    else:
        st.dataframe(
            [
                {
                    "Classe": row["class_name"],
                    "Confiance": f"{row['confidence'] * 100:.1f}%",
                    "x1": round(row["x1"], 1),
                    "y1": round(row["y1"], 1),
                    "x2": round(row["x2"], 1),
                    "y2": round(row["y2"], 1),
                }
                for row in detections
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.download_button(
        label="Télécharger l'image annotée",
        data=annotated_bytes,
        file_name=build_download_filename(last.get("original_name", "image.jpg")),
        mime="image/jpeg",
    )
