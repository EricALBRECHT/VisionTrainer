"""Capture optional training environment metadata (best-effort, never fatal)."""

from __future__ import annotations

import sys
from typing import Any


def collect_training_environment() -> dict[str, Any]:
    """Gather Python / Ultralytics / PyTorch / CUDA info when importable."""
    env: dict[str, Any] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    try:
        import ultralytics

        env["ultralytics"] = getattr(ultralytics, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        env["ultralytics"] = None

    try:
        import torch

        env["torch"] = str(torch.__version__)
        try:
            env["cuda"] = str(getattr(torch.version, "cuda", None) or None)
        except Exception:  # noqa: BLE001
            env["cuda"] = None
    except Exception:  # noqa: BLE001
        env["torch"] = None
        env["cuda"] = None
    return env


def resolve_device_display_name(device: str) -> str | None:
    """Human-readable GPU name for Ultralytics device strings like ``0`` / ``cuda:0``."""
    raw = (device or "").strip().lower()
    if not raw or raw == "cpu":
        return "CPU" if raw == "cpu" else None
    index = 0
    if raw.isdigit():
        index = int(raw)
    elif raw.startswith("cuda:"):
        try:
            index = int(raw.split(":", 1)[1])
        except ValueError:
            index = 0
    elif raw in {"cuda", "0"}:
        index = 0
    else:
        return None
    try:
        from vision_trainer.training.device import get_hardware_info

        info = get_hardware_info()
        for gpu in info.cuda_devices:
            if gpu.index == index:
                return gpu.name
    except Exception:  # noqa: BLE001
        return None
    return None


# Ultralytics default seed (documented); stored when not overridden.
DEFAULT_ULTRALYTICS_SEED = 0
