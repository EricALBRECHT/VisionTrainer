"""Streamlit: video file & camera processing (detect / segment / pipeline)."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from vision_trainer.inference.discovery import discover_trained_models
from vision_trainer.inference.render import ANNOTATION_SCALE_OPTIONS
from vision_trainer.inference.uploads import display_upload_name, safe_internal_upload_path
from vision_trainer.inference.video import (
    SUPPORTED_VIDEO_SUFFIXES,
    VIDEO_DEFAULT_CONF,
    VideoInferenceError,
    build_video_output_filename,
    probe_video,
)
from vision_trainer.pipeline.store import list_pipelines, load_pipeline
from vision_trainer.training.runs import ARTIFACTS_RUNS_DIR
from vision_trainer.ui.device_selector import render_device_selector
from vision_trainer.video.camera import (
    WSL_DOCKER_CAMERA_NOTE,
    CameraError,
    capture_camera_frame,
    list_camera_indices,
)
from vision_trainer.video.engine import (
    process_image_with_processor,
    process_video,
    video_summary_json_bytes,
)
from vision_trainer.video.processors import (
    DetectFrameProcessor,
    PipelineFrameProcessor,
    SegmentFrameProcessor,
)


def render() -> None:
    st.title("Vidéo & Caméra")
    st.markdown(
        """
        Traitement **frame par frame** (détection, segmentation ou pipeline).

        - **FPS vidéo** (lecture / export) ≠ **FPS traitement** (vitesse d'inférence).
        - La vidéo exportée est **sans audio** (limitation OpenCV / V1).
        - Frames sautées (`stride > 1`) : image **originale** (pas d'ancienne annotation).
        """
    )

    st.info(WSL_DOCKER_CAMERA_NOTE)


    @st.cache_resource(show_spinner=False)
    def _load_yolo_model(weights_path: str):
        from ultralytics import YOLO

        return YOLO(weights_path)


    def _model_factory(weights: str):
        return _load_yolo_model(str(weights))


    def _ensure_temp_dir() -> Path:
        from vision_trainer.paths import get_tmp_dir

        try:
            data_tmp = get_tmp_dir()
            if data_tmp.exists():
                key = st.session_state.get("video_temp_dir")
                if key and Path(key).is_dir():
                    return Path(key)
                temp_dir = Path(tempfile.mkdtemp(prefix="vt-vid-", dir=str(data_tmp)))
                st.session_state["video_temp_dir"] = str(temp_dir)
                return temp_dir
        except OSError:
            pass
        temp_dir = Path(st.session_state.get("video_temp_dir") or "")
        if not temp_dir or not temp_dir.exists():
            temp_dir = Path(tempfile.mkdtemp(prefix="vision-trainer-video-"))
            st.session_state["video_temp_dir"] = str(temp_dir)
        return temp_dir


    def _format_duration(seconds: float | None) -> str:
        if seconds is None:
            return "—"
        if seconds < 60:
            return f"{seconds:.1f} s"
        minutes, secs = divmod(int(seconds), 60)
        return f"{minutes} min {secs:02d} s"


    source_kind = st.radio("Source", options=["Vidéo fichier", "Caméra"], horizontal=True)
    mode = st.radio(
        "Mode",
        options=["Détection", "Segmentation", "Pipeline"],
        horizontal=True,
    )
    mode_key = {"Détection": "detect", "Segmentation": "segment", "Pipeline": "pipeline"}[mode]

    device_choice = render_device_selector(key_prefix="video_cam")
    scale_labels = [label for label, _ in ANNOTATION_SCALE_OPTIONS]
    scale_keys = {label: key for label, key in ANNOTATION_SCALE_OPTIONS}
    scale_label = st.selectbox("Taille des annotations", options=scale_labels, index=0)
    annotation_scale = scale_keys[scale_label]

    frame_stride = st.number_input(
        "Traiter 1 frame sur N (stride)",
        min_value=1,
        max_value=30,
        value=1,
        help="Les frames non analysées sont exportées sans annotation (image originale).",
    )
    preview_every = st.number_input(
        "Fréquence preview UI (frames traitées)",
        min_value=1,
        max_value=60,
        value=10,
    )
    export_detailed_json = st.checkbox(
        "Exporter les résultats détaillés frame par frame (JSON volumineux)",
        value=False,
    )

    processor = None
    selected_label = ""

    if mode_key in {"detect", "segment"}:
        models = discover_trained_models(ARTIFACTS_RUNS_DIR, task=mode_key)
        if not models:
            st.warning(f"Aucun modèle `{mode_key}` disponible.")
            st.stop()
        labels = [m.label for m in models]
        selected_label = st.selectbox("Modèle", options=labels)
        selected = next(m for m in models if m.label == selected_label)
        conf = st.slider(
            "Seuil de confiance",
            0.05,
            0.95,
            VIDEO_DEFAULT_CONF if mode_key == "detect" else 0.25,
            0.05,
        )
        iou = st.slider("Seuil IoU", 0.05, 0.95, 0.45, 0.05)
        model_cache: dict = {}
        if mode_key == "detect":
            processor = DetectFrameProcessor(
                weights_path=selected.weights_path,
                conf=float(conf),
                iou=float(iou),
                device_choice=device_choice,
                model_cache=model_cache,
                model_factory=_model_factory,
                annotation_scale=annotation_scale,
            )
        else:
            show_masks = st.checkbox("Masques", value=True)
            show_contours = st.checkbox("Contours", value=True)
            mask_opacity = st.slider("Opacité masque", 0.05, 0.9, 0.4, 0.05)
            processor = SegmentFrameProcessor(
                weights_path=selected.weights_path,
                conf=float(conf),
                iou=float(iou),
                device_choice=device_choice,
                model_cache=model_cache,
                model_factory=_model_factory,
                annotation_scale=annotation_scale,
                mask_opacity=float(mask_opacity),
                show_masks=show_masks,
                show_contours=show_contours,
            )
    else:
        pipelines = list_pipelines()
        if not pipelines:
            st.warning("Aucun pipeline. Créez-en un sur **Pipelines**.")
            st.stop()
        plabels = [f"{p.name} ({p.pipeline_id})" for p in pipelines]
        choice = st.selectbox("Pipeline", options=plabels)
        config = load_pipeline(pipelines[plabels.index(choice)].pipeline_id)
        st.caption(
            f"Détecteur `{config.detector_run_id}` — "
            "seuils classify/segment = configuration pipeline."
        )
        st.warning(
            "Pipeline detect + classify + segment : plus lent et plus gourmand en VRAM "
            "(ex. GTX 1060 3 Go)."
        )
        model_cache = {}
        processor = PipelineFrameProcessor(
            config=config,
            device_choice=device_choice,
            model_cache=model_cache,
            model_factory=_model_factory,
            annotation_scale=annotation_scale,
        )

    assert processor is not None

    # ---------------------------------------------------------------------------
    # Vidéo fichier
    # ---------------------------------------------------------------------------
    if source_kind == "Vidéo fichier":
        uploaded = st.file_uploader(
            "Fichier vidéo",
            type=[s.lstrip(".") for s in sorted(SUPPORTED_VIDEO_SUFFIXES)],
        )
        if uploaded is None:
            st.stop()

        original_name = display_upload_name(uploaded.name)
        raw = uploaded.getvalue()
        fingerprint = hashlib.sha256(raw).hexdigest()
        temp_dir = _ensure_temp_dir()
        video_path = safe_internal_upload_path(temp_dir, original_name)
        video_path.write_bytes(raw)

        try:
            meta = probe_video(video_path)
        except VideoInferenceError as exc:
            st.error(str(exc))
            st.stop()

        st.subheader("Métadonnées")
        st.write(f"- Fichier : `{meta.display_name}`")
        st.write(f"- Résolution : {meta.width or '—'} × {meta.height or '—'}")
        st.write(f"- FPS source : {meta.fps:.2f}" if meta.fps else "- FPS source : —")
        st.write(
            f"- Frames : {meta.frame_count}" if meta.frame_count else "- Frames : —"
        )
        st.write(f"- Durée : {_format_duration(meta.duration_seconds)}")
        st.caption("Audio non conservé dans la vidéo annotée exportée.")

        if st.button("Lancer le traitement vidéo", type="primary"):
            out_name = build_video_output_filename(original_name)
            out_path = video_path.parent / f"out_{mode_key}_{out_name}"
            progress_bar = st.progress(0.0, text="0 %")
            status = st.empty()
            preview_slot = st.empty()

            def _on_progress(progress) -> None:
                if progress.percent is not None:
                    progress_bar.progress(
                        min(1.0, max(0.0, progress.percent / 100.0)),
                        text=f"{progress.percent:.0f} %",
                    )
                parts = [f"Frames lues : {progress.frames_done}"]
                if progress.frames_total:
                    parts[0] = f"Frames : {progress.frames_done} / {progress.frames_total}"
                if progress.processing_fps:
                    parts.append(f"FPS traitement : {progress.processing_fps:.1f}")
                parts.append(f"Temps : {_format_duration(progress.elapsed_seconds)}")
                if progress.eta_seconds is not None:
                    parts.append(f"Reste ≈ {_format_duration(progress.eta_seconds)}")
                status.caption(" · ".join(parts))

            def _on_preview(frame_index: int, image: Image.Image) -> None:
                preview_slot.image(
                    image,
                    caption=f"Preview frame {frame_index}",
                    use_container_width=True,
                )

            try:
                with st.spinner("Traitement vidéo…"):
                    summary = process_video(
                        video_path=video_path,
                        output_path=out_path,
                        processor=processor,
                        mode=mode_key,  # type: ignore[arg-type]
                        frame_stride=int(frame_stride),
                        progress_callback=_on_progress,
                        preview_callback=_on_preview,
                        preview_every=int(preview_every),
                        collect_frame_details=bool(export_detailed_json),
                    )
            except VideoInferenceError as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Erreur : {exc}")
                st.stop()

            st.session_state["video_last_summary"] = {
                "fingerprint": fingerprint,
                "mode": mode_key,
                "summary": summary,
                "out_name": out_name,
            }

        last = st.session_state.get("video_last_summary")
        if last and last.get("fingerprint") == fingerprint and last.get("mode") == mode_key:
            summary = last["summary"]
            st.subheader("Résultat")
            if summary.output_path and Path(summary.output_path).is_file():
                st.video(summary.output_path)
                st.download_button(
                    "Télécharger la vidéo annotée",
                    data=Path(summary.output_path).read_bytes(),
                    file_name=last["out_name"],
                    mime="video/mp4",
                )
            st.write(f"- Frames lues : {summary.frames_read}")
            st.write(f"- Frames traitées : {summary.frames_processed}")
            st.write(f"- Frames sautées : {summary.frames_skipped}")
            st.write(f"- Temps : {_format_duration(summary.elapsed_seconds)}")
            st.write(f"- FPS traitement : {summary.mean_processing_fps:.2f}")
            if summary.mean_ms_per_frame is not None:
                st.write(f"- ms/frame (traitées) : {summary.mean_ms_per_frame:.0f}")
            st.write(f"- FPS export : {summary.output_fps:.2f}")
            st.write(f"- Détections cumulées : {summary.total_detections}")
            if mode_key == "pipeline":
                st.write(f"- Classifications : {summary.total_classifications}")
                st.write(f"- Segmentations (instances) : {summary.total_segmentations}")
                st.write(f"- Erreurs secondaires : {summary.secondary_errors}")
                avg = summary.timings.averages()
                st.caption(
                    "Moyennes pipeline — "
                    f"detect {avg['detection_ms'] or 0:.0f} ms · "
                    f"classify {avg['classification_ms'] or 0:.0f} ms · "
                    f"segment {avg['segmentation_ms'] or 0:.0f} ms"
                )
            for warning in summary.warnings:
                st.warning(warning)
            st.download_button(
                "Télécharger JSON (résumé)",
                data=video_summary_json_bytes(
                    summary, include_frames=bool(export_detailed_json)
                ),
                file_name=f"video_{mode_key}_summary.json",
                mime="application/json",
            )

    # ---------------------------------------------------------------------------
    # Caméra
    # ---------------------------------------------------------------------------
    else:
        st.subheader("Caméra (serveur)")
        max_probe = st.number_input("Indices à sonder", min_value=1, max_value=10, value=3)
        if st.button("Sonder les caméras"):
            infos = list_camera_indices(int(max_probe))
            st.session_state["camera_probe"] = [
                {
                    "index": i.index,
                    "opened": i.opened,
                    "width": i.width,
                    "height": i.height,
                    "message": i.message,
                }
                for i in infos
            ]

        probe = st.session_state.get("camera_probe") or []
        open_cams = [c for c in probe if c.get("opened")]
        if probe and not open_cams:
            st.warning("Aucune caméra accessible depuis le processus serveur.")
        if open_cams:
            st.success(
                "Caméras ouvertes : "
                + ", ".join(f"{c['index']} ({c['width']}×{c['height']})" for c in open_cams)
            )

        cam_index = st.number_input("Index caméra", min_value=0, max_value=9, value=0)

        if st.button("Capturer une frame et analyser", type="primary"):
            try:
                frame = capture_camera_frame(int(cam_index))
            except CameraError as exc:
                st.error(str(exc))
                st.stop()
            st.image(frame, caption=f"Capture caméra {cam_index}", use_container_width=True)
            try:
                with st.spinner("Inférence…"):
                    result = process_image_with_processor(frame, processor)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Erreur : {exc}")
                st.stop()
            if result.annotated is not None:
                st.image(result.annotated, caption="Résultat", use_container_width=True)
            st.write(f"- Détections / instances : {result.detections}")
            st.write(f"- Temps : {result.total_ms:.0f} ms")
            for warning in result.warnings:
                st.warning(warning)
