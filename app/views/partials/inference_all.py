from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from vision_trainer.classify.predictor import run_classify_inference
from vision_trainer.inference.discovery import discover_trained_models
from vision_trainer.inference.predictor import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    InferenceError,
    load_image_rgb,
    run_inference,
)
from vision_trainer.inference.render import (
    ANNOTATION_SCALE_OPTIONS,
    annotated_image_to_jpeg_bytes,
    build_download_filename,
    draw_detections,
)
from vision_trainer.inference.uploads import (
    SUPPORTED_UPLOAD_SUFFIXES,
    SUPPORTED_VIDEO_UPLOAD_SUFFIXES,
    display_upload_name,
    safe_internal_upload_path,
)
from vision_trainer.inference.video import (
    VIDEO_DEFAULT_CONF,
    VideoInferenceError,
    build_video_output_filename,
    probe_video,
    run_video_inference,
)
from vision_trainer.results.models import SESSION_INFERENCE_WEIGHTS_KEY
from vision_trainer.segment.export import segmentation_result_json_bytes
from vision_trainer.segment.predictor import result_table_rows as segment_table_rows
from vision_trainer.segment.predictor import run_segmentation
from vision_trainer.segment.render import draw_segmentation_result
from vision_trainer.training.runs import ARTIFACTS_RUNS_DIR
from vision_trainer.ui.device_selector import render_device_selector

from views.partials.inference_display import (
    render_inference_display_size_selector,
    show_inference_preview,
)


def render(*, task_key: str | None = None) -> None:
    """Render inference UI.

    Parameters
    ----------
    task_key:
        Optional ``detect`` / ``classify`` / ``segment``. When omitted, a radio
        selector is shown (legacy / standalone use).
    """
    label_by_key = {
        "detect": "Détection",
        "classify": "Classification",
        "segment": "Segmentation",
    }
    if task_key is None:
        infer_task = st.radio(
            "Type de modèle",
            options=["Détection", "Classification", "Segmentation"],
            horizontal=True,
            key="inference_task_mode",
        )
        task_key = {
            "Détection": "detect",
            "Classification": "classify",
            "Segmentation": "segment",
        }[infer_task]
    else:
        if task_key not in label_by_key:
            raise ValueError(f"task_key invalide: {task_key!r}")
        infer_task = label_by_key[task_key]
        # Keep legacy session key in sync for deep-links from Résultats.
        st.session_state["inference_task_mode"] = infer_task

    models = discover_trained_models(ARTIFACTS_RUNS_DIR, task=task_key)
    if not models:
        st.warning(
            f"Aucun modèle de **{infer_task.lower()}** disponible (`weights/best.pt`). "
            "Lancez un entraînement correspondant d'abord."
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

    display_fraction = render_inference_display_size_selector(key="inference_display_size")


    @st.cache_resource(show_spinner=False)
    def _load_yolo_model(weights_path: str):
        from ultralytics import YOLO

        return YOLO(weights_path)


    def _ensure_temp_dir() -> Path:
        from vision_trainer.paths import get_tmp_dir

        # Prefer persistent data/tmp when configured (Docker), else session temp.
        try:
            data_tmp = get_tmp_dir()
            if data_tmp.exists():
                session_key = st.session_state.get("inference_temp_dir")
                if session_key and Path(session_key).is_dir():
                    return Path(session_key)
                temp_dir = Path(tempfile.mkdtemp(prefix="vt-inf-", dir=str(data_tmp)))
                st.session_state["inference_temp_dir"] = str(temp_dir)
                return temp_dir
        except OSError:
            pass

        temp_dir = Path(st.session_state.get("inference_temp_dir") or "")
        if not temp_dir or not temp_dir.exists():
            temp_dir = Path(tempfile.mkdtemp(prefix="vision-trainer-inference-"))
            st.session_state["inference_temp_dir"] = str(temp_dir)
        return temp_dir


    def _format_duration(seconds: float | None) -> str:
        if seconds is None:
            return "—"
        if seconds < 60:
            return f"{seconds:.1f} s"
        minutes, secs = divmod(int(seconds), 60)
        return f"{minutes} min {secs:02d} s"


    media_mode = "Image"
    if task_key == "detect":
        media_mode = st.radio("Type de média", options=["Image", "Vidéo"], horizontal=True)

    st.subheader("Paramètres")

    if task_key == "classify":
        min_confidence = st.slider(
            "Seuil de confiance minimum",
            min_value=0.05,
            max_value=0.99,
            value=0.80,
            step=0.05,
            help=(
                "Si le score Top-1 est inférieur à ce seuil, le résultat est "
                "INCONNU / CONFIANCE INSUFFISANTE. Ce n'est pas une détection OOD formelle."
            ),
            key="cls_min_confidence",
        )
        min_margin = st.slider(
            "Écart minimum Top-1 / Top-2",
            min_value=0.0,
            max_value=0.50,
            value=0.10,
            step=0.01,
            help="Si Top-1 − Top-2 est inférieur à cet écart, le résultat est INCERTAIN.",
            key="cls_min_margin",
        )
        top_n = st.slider(
            "Top-N",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            key="cls_top_n",
        )
        st.caption(
            "Le seuil et l'écart Top-1/Top-2 sont des garde-fous simples. "
            "Ils ne garantissent pas de détecter toutes les images hors distribution (OOD)."
        )
        device_choice, resolved_device, _resolved_label = render_device_selector(
            key_prefix="infer_classify",
            default_choice="auto",
        )

        uploaded = st.file_uploader(
            "Image à classifier",
            type=["jpg", "jpeg", "png", "webp"],
            help="Formats acceptés : JPG, JPEG, PNG, WEBP.",
        )
        if uploaded is None:
            st.info("Chargez une image pour tester le classificateur.")
            st.stop()

        original_name = display_upload_name(uploaded.name)
        suffix = Path(original_name).suffix.lower()
        if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
            st.error("Format d'image non supporté.")
            st.stop()

        temp_dir = _ensure_temp_dir()
        image_path = safe_internal_upload_path(temp_dir, original_name)
        image_path.write_bytes(uploaded.getvalue())
        try:
            original_image = load_image_rgb(image_path)
        except InferenceError as exc:
            st.error(str(exc))
            st.stop()

        show_inference_preview(
            original_image,
            display_fraction=display_fraction,
            caption=original_name,
        )

        if st.button("Lancer l'inférence", type="primary"):
            try:
                with st.spinner("Classification…"):
                    cached_model = _load_yolo_model(selected_model.weights_path)

                    def _factory(_path: str):
                        return cached_model

                    result = run_classify_inference(
                        weights_path=selected_model.weights_path,
                        image=original_image,
                        device_choice=device_choice,
                        min_confidence=float(min_confidence),
                        min_margin=float(min_margin),
                        top_n=int(top_n),
                        model_factory=_factory,
                    )
            except InferenceError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Erreur inattendue : {exc}")
            else:
                decision = result.decision
                st.subheader("Résultat")
                if decision.kind == "accepted":
                    st.success(
                        f"**Classe principale :** {decision.label} — "
                        f"{decision.top1.confidence * 100:.1f} %"
                        if decision.top1
                        else f"**Classe principale :** {decision.label}"
                    )
                elif decision.kind == "uncertain":
                    st.warning(f"**{decision.label}**")
                    if decision.reason:
                        st.caption(decision.reason)
                else:
                    st.error(f"**{decision.label}**")
                    if decision.reason:
                        st.caption(decision.reason)

                st.markdown("#### Top-N")
                rows = [
                    {
                        "Classe": score.class_name,
                        "Score": f"{score.confidence * 100:.1f} %",
                    }
                    for score in decision.top_n
                ]
                if rows:
                    st.dataframe(rows, use_container_width=True, hide_index=True)
                else:
                    st.info("Aucune probabilité disponible.")
        st.stop()

    # ---------------------------------------------------------------------------
    # Segmentation (image)
    # ---------------------------------------------------------------------------
    if task_key == "segment":
        conf = st.slider(
            "Seuil de confiance",
            min_value=0.05,
            max_value=0.95,
            value=DEFAULT_CONF,
            step=0.05,
            key="seg_inference_conf",
        )
        iou = st.slider(
            "Seuil IoU",
            min_value=0.05,
            max_value=0.95,
            value=DEFAULT_IOU,
            step=0.05,
            key="seg_inference_iou",
        )
        show_masks = st.checkbox("Afficher masques", value=True, key="seg_show_masks")
        show_contours = st.checkbox("Afficher contours", value=True, key="seg_show_contours")
        show_labels = st.checkbox("Afficher labels", value=True, key="seg_show_labels")
        show_boxes = st.checkbox("Afficher bounding boxes", value=False, key="seg_show_boxes")
        mask_opacity = st.slider(
            "Opacité du masque",
            min_value=0.05,
            max_value=0.90,
            value=0.40,
            step=0.05,
            key="seg_mask_opacity",
        )
        annotation_label = st.selectbox(
            "Taille des annotations",
            options=[label for label, _ in ANNOTATION_SCALE_OPTIONS],
            index=0,
            key="seg_annotation_scale",
        )
        annotation_scale = next(
            key for label, key in ANNOTATION_SCALE_OPTIONS if label == annotation_label
        )
        device_choice, resolved_device, _resolved_label = render_device_selector(
            key_prefix="infer_segment",
            default_choice="auto",
        )

        uploaded = st.file_uploader(
            "Image à segmenter",
            type=["jpg", "jpeg", "png", "webp"],
            help="Formats acceptés : JPG, JPEG, PNG, WEBP.",
        )
        if uploaded is None:
            st.info("Chargez une image pour tester le modèle de segmentation.")
            st.stop()

        original_name = display_upload_name(uploaded.name)
        suffix = Path(original_name).suffix.lower()
        if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
            st.error("Format d'image non supporté.")
            st.stop()

        temp_dir = _ensure_temp_dir()
        image_path = safe_internal_upload_path(temp_dir, original_name)
        image_path.write_bytes(uploaded.getvalue())
        try:
            original_image = load_image_rgb(image_path)
        except InferenceError as exc:
            st.error(str(exc))
            st.stop()

        show_inference_preview(
            original_image,
            display_fraction=display_fraction,
            caption=original_name,
        )

        if st.button("Lancer l'inférence", type="primary", key="seg_run"):
            try:
                with st.spinner("Segmentation…"):
                    cached_model = _load_yolo_model(selected_model.weights_path)

                    def _factory(_path: str):
                        return cached_model

                    result = run_segmentation(
                        weights_path=selected_model.weights_path,
                        image=original_image,
                        conf=float(conf),
                        iou=float(iou),
                        device_choice=device_choice,
                        model_factory=_factory,
                    )
            except InferenceError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Erreur inattendue : {exc}")
            else:
                annotated = draw_segmentation_result(
                    original_image,
                    result,
                    show_masks=show_masks,
                    show_contours=show_contours,
                    show_labels=show_labels,
                    show_boxes=show_boxes,
                    mask_opacity=float(mask_opacity),
                    scale=annotation_scale,
                )
                st.subheader("Résultat")
                show_inference_preview(
                    annotated,
                    display_fraction=display_fraction,
                    caption="Segmentation",
                )
                st.download_button(
                    "Télécharger l'image annotée",
                    data=annotated_image_to_jpeg_bytes(annotated),
                    file_name=build_download_filename(original_name),
                    mime="image/jpeg",
                    key="seg_dl_image",
                )
                st.download_button(
                    "Exporter JSON (polygones / surfaces)",
                    data=segmentation_result_json_bytes(result),
                    file_name=f"{Path(original_name).stem}_segmentation.json",
                    mime="application/json",
                    key="seg_dl_json",
                )
                st.caption(
                    "La surface est en **pixels** (et ratio d'image), pas une mesure physique."
                )
                rows = segment_table_rows(result)
                if rows:
                    st.dataframe(rows, use_container_width=True, hide_index=True)
                else:
                    st.info("Aucune instance détectée.")
        st.stop()

    # ---------------------------------------------------------------------------
    # Détection (image / vidéo) — comportement historique
    # ---------------------------------------------------------------------------
    default_conf = DEFAULT_CONF if media_mode == "Image" else VIDEO_DEFAULT_CONF
    conf = st.slider(
        "Seuil de confiance",
        min_value=0.05,
        max_value=0.95,
        value=default_conf,
        step=0.05,
        key=f"inference_conf_{media_mode}",
    )
    iou = st.slider(
        "Seuil IoU",
        min_value=0.05,
        max_value=0.95,
        value=DEFAULT_IOU,
        step=0.05,
        key=f"inference_iou_{media_mode}",
    )
    annotation_label = st.selectbox(
        "Taille des annotations",
        options=[label for label, _ in ANNOTATION_SCALE_OPTIONS],
        index=0,
        help=(
            "Auto adapte l'épaisseur des boîtes et la taille du texte à la résolution. "
            "Petite / Moyenne / Grande forcent un rendu plus compact ou plus lisible."
        ),
        key=f"inference_annotation_scale_{media_mode}",
    )
    annotation_scale = next(
        key for label, key in ANNOTATION_SCALE_OPTIONS if label == annotation_label
    )

    device_choice, resolved_device, _resolved_label = render_device_selector(
        key_prefix=f"infer_{media_mode}",
        default_choice="auto",
    )

    # ---------------------------------------------------------------------------
    # Image mode (unchanged behaviour)
    # ---------------------------------------------------------------------------
    if media_mode == "Image":
        uploaded = st.file_uploader(
            "Image à analyser",
            type=["jpg", "jpeg", "png", "webp"],
            help="Formats acceptés : JPG, JPEG, PNG, WEBP.",
        )

        original_image: Image.Image | None = None
        original_name = "image.jpg"
        image_fingerprint = ""

        if uploaded is not None:
            original_name = display_upload_name(uploaded.name)
            suffix = Path(original_name).suffix.lower()
            if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
                st.error("Format d'image non supporté.")
                st.stop()

            temp_dir = _ensure_temp_dir()
            raw_bytes = uploaded.getvalue()
            image_fingerprint = hashlib.sha256(raw_bytes).hexdigest()
            image_path = safe_internal_upload_path(temp_dir, original_name)
            image_path.write_bytes(raw_bytes)

            try:
                original_image = load_image_rgb(image_path)
            except InferenceError as exc:
                st.error(str(exc))
                st.stop()

            st.subheader("Image originale")
            show_inference_preview(
                original_image,
                display_fraction=display_fraction,
                caption=original_name,
            )

        current_binding = {
            "mode": "image",
            "image_sha256": image_fingerprint,
            "weights": str(Path(selected_model.weights_path).resolve()),
            "conf": float(conf),
            "iou": float(iou),
            "device": device_choice,
            "annotation_scale": annotation_scale,
        }

        can_run = original_image is not None and bool(selected_model.weights_path)
        run_clicked = st.button("Lancer l'inférence", type="primary", disabled=not can_run)

        if run_clicked:
            if original_image is None:
                st.error("Chargez une image avant de lancer l'inférence.")
                st.stop()

            try:
                with st.spinner("Inférence en cours…"):
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
                    annotated = draw_detections(
                        original_image,
                        result.detections,
                        scale=annotation_scale,
                    )
            except InferenceError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Erreur inattendue pendant l'inférence : {exc}")
            else:
                st.session_state["inference_last_result"] = {
                    "binding": current_binding,
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

        last = st.session_state.get("inference_last_result")
        if last and last.get("binding") == current_binding and current_binding["image_sha256"]:
            st.subheader("Résultat")
            detections = last["detections"]
            annotated_bytes = last["annotated_jpeg"]

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("**Originale**")
                if original_image is not None:
                    show_inference_preview(
                        original_image,
                        display_fraction=display_fraction,
                    )
                else:
                    st.caption("Rechargez une image pour revoir l'originale à côté du résultat.")
            with col_right:
                st.markdown("**Annotée**")
                show_inference_preview(
                    annotated_bytes,
                    display_fraction=display_fraction,
                )

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
        elif last and last.get("binding") != current_binding:
            st.caption(
                "Un résultat précédent existe mais ne correspond plus à l'image / modèle / "
                "seuils / device actuels. Relancez l'inférence."
            )

    # ---------------------------------------------------------------------------
    # Video mode
    # ---------------------------------------------------------------------------
    else:
        uploaded_video = st.file_uploader(
            "Vidéo à analyser",
            type=["mp4", "avi", "mov", "mkv"],
            help="Formats acceptés : MP4, AVI, MOV, MKV.",
        )

        video_path: Path | None = None
        video_name = "video.mp4"
        video_fingerprint = ""
        video_meta = None

        if uploaded_video is not None:
            video_name = display_upload_name(uploaded_video.name, default_name="video.mp4")
            suffix = Path(video_name).suffix.lower()
            if suffix not in SUPPORTED_VIDEO_UPLOAD_SUFFIXES:
                st.error("Format vidéo non supporté.")
                st.stop()

            temp_dir = _ensure_temp_dir()
            raw_bytes = uploaded_video.getvalue()
            video_fingerprint = hashlib.sha256(raw_bytes).hexdigest()
            video_path = safe_internal_upload_path(
                temp_dir,
                video_name,
                allowed_suffixes=SUPPORTED_VIDEO_UPLOAD_SUFFIXES,
                default_suffix=".mp4",
            )
            video_path.write_bytes(raw_bytes)

            try:
                video_meta = probe_video(video_path)
            except VideoInferenceError as exc:
                st.error(str(exc))
                st.stop()

            st.subheader("Vidéo source")
            st.write(f"**Fichier :** `{video_name}`")
            st.write(
                f"- Résolution : "
                f"{video_meta.width or '—'} × {video_meta.height or '—'}"
            )
            st.write(f"- FPS : {video_meta.fps:.2f}" if video_meta.fps else "- FPS : —")
            st.write(
                f"- Frames (approx.) : {video_meta.frame_count}"
                if video_meta.frame_count
                else "- Frames (approx.) : —"
            )
            st.write(f"- Durée : {_format_duration(video_meta.duration_seconds)}")
            st.video(str(video_path))

        current_binding = {
            "mode": "video",
            "video_sha256": video_fingerprint,
            "weights": str(Path(selected_model.weights_path).resolve()),
            "conf": float(conf),
            "iou": float(iou),
            "device": device_choice,
            "annotation_scale": annotation_scale,
        }

        can_run = video_path is not None and bool(selected_model.weights_path)
        run_clicked = st.button("Lancer l'inférence", type="primary", disabled=not can_run)

        if run_clicked:
            if video_path is None:
                st.error("Chargez une vidéo avant de lancer l'inférence.")
                st.stop()

            output_name = build_video_output_filename(video_name)
            output_path = video_path.parent / f"out_{output_name}"

            progress_bar = st.progress(0.0, text="Traitement vidéo — 0 %")
            status_box = st.empty()

            def _on_progress(progress) -> None:
                if progress.percent is not None:
                    progress_bar.progress(
                        min(1.0, max(0.0, progress.percent / 100.0)),
                        text=f"Traitement vidéo — {progress.percent:.0f} %",
                    )
                else:
                    progress_bar.progress(0.0, text="Traitement vidéo — en cours…")

                parts = [f"Frames : {progress.frames_done}"]
                if progress.frames_total:
                    parts[0] = f"Frames : {progress.frames_done} / {progress.frames_total}"
                if progress.processing_fps:
                    parts.append(f"FPS traitement : {progress.processing_fps:.1f}")
                parts.append(f"Temps : {_format_duration(progress.elapsed_seconds)}")
                if progress.eta_seconds is not None:
                    parts.append(f"Reste ≈ {_format_duration(progress.eta_seconds)}")
                status_box.caption(" · ".join(parts))

            try:
                cached_model = _load_yolo_model(selected_model.weights_path)

                def _factory(_path: str):
                    return cached_model

                video_result = run_video_inference(
                    weights_path=selected_model.weights_path,
                    video_path=video_path,
                    output_path=output_path,
                    conf=float(conf),
                    iou=float(iou),
                    device_choice=device_choice,
                    model_factory=_factory,
                    progress_callback=_on_progress,
                    annotation_scale=annotation_scale,
                )
            except VideoInferenceError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Erreur inattendue pendant l'inférence vidéo : {exc}")
            else:
                progress_bar.progress(1.0, text="Traitement vidéo — 100 %")
                st.session_state["inference_last_video_result"] = {
                    "binding": current_binding,
                    "output_path": video_result.output_path,
                    "output_name": output_name,
                    "frames_analyzed": video_result.frames_analyzed,
                    "elapsed_seconds": video_result.elapsed_seconds,
                    "mean_processing_fps": video_result.mean_processing_fps,
                    "total_detections": video_result.total_detections,
                    "detections_by_class": video_result.detections_by_class,
                    "source_width": video_result.source_width,
                    "source_height": video_result.source_height,
                    "output_fps": video_result.output_fps,
                }

        last_video = st.session_state.get("inference_last_video_result")
        if (
            last_video
            and last_video.get("binding") == current_binding
            and current_binding["video_sha256"]
            and Path(last_video.get("output_path", "")).is_file()
        ):
            st.subheader("Résultat")
            st.caption(
                "Les totaux par classe sont des détections **cumulées sur les frames** "
                "(pas d'objets uniques — aucun tracking)."
            )
            st.video(last_video["output_path"])

            st.write(f"- Frames analysées : {last_video['frames_analyzed']:,}".replace(",", " "))
            st.write(f"- Temps : {_format_duration(last_video['elapsed_seconds'])}")
            st.write(f"- FPS moyen de traitement : {last_video['mean_processing_fps']:.1f}")
            st.write(f"- Détections cumulées : {last_video['total_detections']:,}".replace(",", " "))
            by_class = last_video.get("detections_by_class") or {}
            if by_class:
                for class_name, count in by_class.items():
                    st.write(f"- {class_name} : {count:,} détections".replace(",", " "))
            else:
                st.info("Aucune détection au-dessus du seuil sélectionné.")

            try:
                video_bytes = Path(last_video["output_path"]).read_bytes()
            except OSError as exc:
                st.warning(f"Téléchargement indisponible : {exc}")
            else:
                st.download_button(
                    label="Télécharger la vidéo annotée",
                    data=video_bytes,
                    file_name=last_video.get("output_name", "video_inference.mp4"),
                    mime="video/mp4",
                )
        elif last_video and last_video.get("binding") != current_binding:
            st.caption(
                "Un résultat vidéo précédent existe mais ne correspond plus à la vidéo / modèle / "
                "seuils / device actuels. Relancez l'inférence."
            )
