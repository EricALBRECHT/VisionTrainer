from __future__ import annotations

from typing import Literal

DeviceChoice = Literal["auto", "cpu", "cuda"]


class DeviceError(Exception):
    """Raised when the requested training device cannot be used."""


def is_cuda_available() -> bool:
    """Return True when a CUDA device is available via torch."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def resolve_device(choice: DeviceChoice | str) -> str:
    """
    Resolve the Ultralytics ``device`` argument.

    - ``auto`` → ``0`` if CUDA is available, else ``cpu``
    - ``cpu`` → ``cpu``
    - ``cuda`` → ``0`` if CUDA is available, else raises ``DeviceError``
    """
    normalized = str(choice).strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise DeviceError(f"Choix de device invalide : {choice!r}.")

    cuda_ok = is_cuda_available()

    if normalized == "cpu":
        return "cpu"

    if normalized == "cuda":
        if not cuda_ok:
            raise DeviceError(
                "CUDA a été demandé mais aucun GPU CUDA n'est disponible."
            )
        return "0"

    # auto
    return "0" if cuda_ok else "cpu"


def describe_device(device: str) -> str:
    """Human-readable label for the resolved Ultralytics device value."""
    if device == "cpu":
        return "CPU"
    if device in {"0", "cuda", "cuda:0"}:
        return "CUDA (GPU 0)"
    return str(device)
