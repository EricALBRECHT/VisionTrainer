"""Segmentation inference engine (Streamlit-independent)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from vision_trainer.inference.predictor import (
    InferenceError,
    extract_class_names,
    load_image_rgb,
)
from vision_trainer.segment.area import mask_area_pixels, mask_area_ratio, polygon_area_pixels
from vision_trainer.segment.models import SegmentInstance, SegmentationResult
from vision_trainer.training.device import DeviceChoice, DeviceError, resolve_device

ModelFactory = Callable[[str], Any]


def run_segmentation(
    *,
    weights_path: Path | str,
    image: Image.Image | Path | str,
    conf: float = 0.25,
    iou: float = 0.45,
    device_choice: DeviceChoice | str = "auto",
    model_factory: ModelFactory | None = None,
) -> SegmentationResult:
    """
    Run a YOLO segmentation model and return structured instances.

    Inference uses the original image. Rendering is a separate concern.
    """
    weights = Path(weights_path)
    if not weights.is_file():
        raise InferenceError(f"Poids introuvables : {weights}")

    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        pil = load_image_rgb(image)

    try:
        device = resolve_device(device_choice)
    except DeviceError as exc:
        raise InferenceError(str(exc)) from exc

    factory = model_factory or _default_factory
    try:
        model = factory(str(weights))
    except Exception as exc:  # noqa: BLE001
        raise InferenceError(f"Impossible de charger le modèle : {exc}") from exc

    class_names = extract_class_names(model)

    try:
        results = model.predict(
            source=pil,
            conf=float(conf),
            iou=float(iou),
            device=device,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise InferenceError(f"Erreur pendant l'inférence segmentation : {exc}") from exc

    return normalize_segmentation_results(
        results,
        class_names=class_names,
        image_width=pil.width,
        image_height=pil.height,
        device=device,
    )


def normalize_segmentation_results(
    results: Any,
    *,
    class_names: dict[int, str] | None = None,
    image_width: int = 0,
    image_height: int = 0,
    device: str = "cpu",
) -> SegmentationResult:
    """Normalize Ultralytics Results into SegmentInstance list."""
    names = dict(class_names or {})
    items = results if isinstance(results, (list, tuple)) else [results]
    instances: list[SegmentInstance] = []
    width = image_width
    height = image_height

    for result in items:
        if result is None:
            continue
        orig = getattr(result, "orig_shape", None)
        if orig is not None and len(orig) >= 2:
            # Ultralytics orig_shape is (h, w)
            height = int(orig[0]) or height
            width = int(orig[1]) or width

        result_names = getattr(result, "names", None)
        if isinstance(result_names, dict) and not names:
            names = extract_class_names(type("M", (), {"names": result_names})())

        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)

        if boxes is None:
            continue

        xyxy = _to_list2d(getattr(boxes, "xyxy", None))
        confs = _to_list1d(getattr(boxes, "conf", None))
        clss = _to_list1d(getattr(boxes, "cls", None))
        count = max(len(xyxy), len(confs), len(clss))

        polygons = _extract_polygons(masks)
        mask_arrays = _extract_mask_arrays(masks)

        for index in range(count):
            if index >= len(xyxy) or len(xyxy[index]) < 4:
                continue
            x1, y1, x2, y2 = (float(v) for v in xyxy[index][:4])
            confidence = float(confs[index]) if index < len(confs) else 0.0
            class_id = int(clss[index]) if index < len(clss) else -1
            class_name = names.get(class_id, f"class_{class_id}")

            polygon: tuple[tuple[float, float], ...] = ()
            if index < len(polygons):
                polygon = polygons[index]

            area = 0
            if index < len(mask_arrays) and mask_arrays[index] is not None:
                try:
                    # Resize-free: mask may be at model size — prefer polygon area
                    # when dimensions mismatch original image.
                    mask = mask_arrays[index]
                    if mask.shape[0] == height and mask.shape[1] == width:
                        area = mask_area_pixels(mask)
                    elif polygon:
                        area = polygon_area_pixels(
                            polygon, image_width=width, image_height=height
                        )
                    else:
                        # Approximate: scale ratio of nonzero on mask resolution
                        area = mask_area_pixels(mask)
                        # Remap proportionally to original if shapes differ
                        mh, mw = mask.shape[:2]
                        if mh > 0 and mw > 0 and (mh != height or mw != width):
                            ratio = area / float(mh * mw)
                            area = int(round(ratio * width * height))
                except Exception:  # noqa: BLE001
                    area = 0
            elif polygon:
                area = polygon_area_pixels(
                    polygon, image_width=width, image_height=height
                )

            ratio = mask_area_ratio(area, image_width=width, image_height=height)
            instances.append(
                SegmentInstance(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    polygon=polygon,
                    mask_area_pixels=area,
                    mask_area_ratio=ratio,
                )
            )

    return SegmentationResult(
        instances=instances,
        class_names=names,
        image_width=width,
        image_height=height,
        device=device,
    )


def _extract_polygons(masks: Any) -> list[tuple[tuple[float, float], ...]]:
    if masks is None:
        return []
    xy = getattr(masks, "xy", None)
    if xy is None:
        return []
    polygons: list[tuple[tuple[float, float], ...]] = []
    try:
        for poly in xy:
            if poly is None:
                polygons.append(())
                continue
            arr = np.asarray(poly)
            if arr.size == 0:
                polygons.append(())
                continue
            if arr.ndim != 2 or arr.shape[1] < 2:
                polygons.append(())
                continue
            points = tuple((float(row[0]), float(row[1])) for row in arr)
            polygons.append(points)
    except Exception:  # noqa: BLE001
        return []
    return polygons


def _extract_mask_arrays(masks: Any) -> list[np.ndarray | None]:
    if masks is None:
        return []
    data = getattr(masks, "data", None)
    if data is None:
        return []
    try:
        array = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
    except Exception:  # noqa: BLE001
        return []
    if array.ndim == 2:
        return [array]
    if array.ndim == 3:
        return [array[i] for i in range(array.shape[0])]
    return []


def _to_list2d(value: Any) -> list[list[float]]:
    if value is None:
        return []
    try:
        if hasattr(value, "cpu"):
            value = value.cpu().numpy()
        array = np.asarray(value)
        return array.tolist()
    except Exception:  # noqa: BLE001
        return []


def _to_list1d(value: Any) -> list[float]:
    if value is None:
        return []
    try:
        if hasattr(value, "cpu"):
            value = value.cpu().numpy()
        array = np.asarray(value).reshape(-1)
        return [float(v) for v in array.tolist()]
    except Exception:  # noqa: BLE001
        return []


def _default_factory(weights: str) -> Any:
    from ultralytics import YOLO

    return YOLO(weights)


def result_table_rows(result: SegmentationResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in result.instances:
        rows.append(
            {
                "Classe": item.class_name,
                "Confiance": round(item.confidence, 4),
                "Surface masque (px)": item.mask_area_pixels,
                "Ratio image": round(item.mask_area_ratio, 6),
                "BBox": f"{item.x1:.0f},{item.y1:.0f},{item.x2:.0f},{item.y2:.0f}",
            }
        )
    return rows
