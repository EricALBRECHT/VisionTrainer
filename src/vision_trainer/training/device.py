from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

DeviceChoice = Literal["auto", "cpu", "cuda"]


class DeviceError(Exception):
    """Raised when the requested training / inference device cannot be used."""


@dataclass(frozen=True)
class CudaDeviceInfo:
    """One physical CUDA device."""

    index: int
    name: str
    total_memory_bytes: int | None = None

    @property
    def label(self) -> str:
        return f"GPU — {self.name}"

    @property
    def ultralytics_device(self) -> str:
        """Ultralytics / torch device string for this GPU."""
        return str(self.index)


@dataclass(frozen=True)
class HardwareInfo:
    """Snapshot of host/container compute capabilities."""

    cuda_available: bool
    device_count: int
    cuda_devices: tuple[CudaDeviceInfo, ...] = ()
    torch_version: str | None = None
    cuda_version: str | None = None
    cudnn_version: str | None = None

    @property
    def primary_gpu(self) -> CudaDeviceInfo | None:
        return self.cuda_devices[0] if self.cuda_devices else None


@dataclass(frozen=True)
class DeviceOption:
    """One selectable entry for Streamlit device radios / selects."""

    choice: DeviceChoice
    label: str
    """Value passed to ``resolve_device`` (auto / cpu / cuda)."""


def is_cuda_available() -> bool:
    """Return True when a CUDA device is available via torch."""
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 — broken CUDA stacks must not crash the UI
        return False


def get_hardware_info() -> HardwareInfo:
    """Collect CPU/GPU hardware details for diagnostics (safe without CUDA)."""
    torch_version = None
    cuda_version = None
    cudnn_version = None
    devices: list[CudaDeviceInfo] = []
    cuda_ok = False
    count = 0

    try:
        import torch
    except ImportError:
        return HardwareInfo(
            cuda_available=False,
            device_count=0,
            cuda_devices=(),
            torch_version=None,
            cuda_version=None,
            cudnn_version=None,
        )

    torch_version = str(torch.__version__)
    try:
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        cuda_ok = False

    if cuda_ok:
        try:
            count = int(torch.cuda.device_count())
        except Exception:  # noqa: BLE001
            count = 0
        cuda_version = getattr(torch.version, "cuda", None)
        try:
            if torch.backends.cudnn.is_available():
                cudnn_version = str(torch.backends.cudnn.version())
        except Exception:  # noqa: BLE001
            cudnn_version = None

        for index in range(count):
            name = f"CUDA:{index}"
            memory = None
            try:
                name = str(torch.cuda.get_device_name(index))
            except Exception:  # noqa: BLE001
                pass
            try:
                props = torch.cuda.get_device_properties(index)
                memory = int(getattr(props, "total_memory", 0)) or None
            except Exception:  # noqa: BLE001
                memory = None
            devices.append(
                CudaDeviceInfo(index=index, name=name, total_memory_bytes=memory)
            )

    return HardwareInfo(
        cuda_available=cuda_ok and count > 0,
        device_count=count if cuda_ok else 0,
        cuda_devices=tuple(devices),
        torch_version=torch_version,
        cuda_version=str(cuda_version) if cuda_version else None,
        cudnn_version=cudnn_version,
    )


def get_available_devices() -> list[DeviceOption]:
    """
    Build UI options for device selection.

    Always includes Auto and CPU. Adds one GPU option per CUDA device when available
    (selection still maps to ``cuda`` / first GPU for Ultralytics simplicity; multi-GPU
    index is encoded in the label for display, resolve uses GPU 0 via ``cuda``).
    """
    info = get_hardware_info()
    options: list[DeviceOption] = [
        DeviceOption(choice="auto", label="Auto"),
        DeviceOption(choice="cpu", label="CPU"),
    ]
    if info.cuda_available:
        for gpu in info.cuda_devices:
            # Map every listed GPU to choice "cuda" for the first card; for index > 0
            # we still expose labels but resolve_device("cuda") uses device 0 unless
            # the choice string is "cuda:N" — support cuda:N below.
            choice: DeviceChoice = "cuda"
            label = gpu.label if gpu.index == 0 else f"GPU {gpu.index} — {gpu.name}"
            options.append(DeviceOption(choice=choice, label=label))
            # Only expose a single "cuda" choice entry for Streamlit uniqueness when
            # multiple GPUs share the same choice key — keep first GPU only for V1.
            break
    return options


def resolve_device(choice: DeviceChoice | str) -> str:
    """
    Resolve the Ultralytics ``device`` argument.

    - ``auto`` → ``0`` if CUDA is available, else ``cpu``
    - ``cpu`` → ``cpu``
    - ``cuda`` / ``cuda:0`` / ``0`` → first GPU when available
    - ``cuda:N`` / ``N`` → GPU N when available
    """
    raw = str(choice).strip().lower()
    if not raw:
        raise DeviceError("Choix de device vide.")

    cuda_ok = is_cuda_available()

    if raw == "cpu":
        return "cpu"

    if raw == "auto":
        return "0" if cuda_ok else "cpu"

    # Explicit CUDA request
    gpu_index: int | None = None
    if raw in {"cuda", "gpu"}:
        gpu_index = 0
    elif raw.startswith("cuda:"):
        try:
            gpu_index = int(raw.split(":", 1)[1])
        except ValueError as exc:
            raise DeviceError(f"Choix de device invalide : {choice!r}.") from exc
    elif raw.isdigit():
        gpu_index = int(raw)
    else:
        raise DeviceError(f"Choix de device invalide : {choice!r}.")

    if not cuda_ok:
        raise DeviceError(
            "CUDA a été demandé mais aucun GPU CUDA n'est disponible."
        )

    try:
        import torch

        count = int(torch.cuda.device_count())
    except Exception as exc:  # noqa: BLE001
        raise DeviceError(
            "CUDA a été demandé mais aucun GPU CUDA n'est disponible."
        ) from exc

    if gpu_index < 0 or gpu_index >= count:
        raise DeviceError(
            f"GPU {gpu_index} demandé mais seulement {count} device(s) CUDA "
            f"disponible(s)."
        )
    return str(gpu_index)


def describe_device(device: str, *, hardware: HardwareInfo | None = None) -> str:
    """Human-readable label for the resolved Ultralytics device value."""
    if device == "cpu":
        return "CPU"

    info = hardware if hardware is not None else get_hardware_info()
    index: int | None = None
    text = str(device).strip().lower()
    if text in {"0", "cuda", "cuda:0", "gpu"}:
        index = 0
    elif text.startswith("cuda:"):
        try:
            index = int(text.split(":", 1)[1])
        except ValueError:
            index = None
    elif text.isdigit():
        index = int(text)

    if index is not None:
        for gpu in info.cuda_devices:
            if gpu.index == index:
                return f"GPU — {gpu.name}"
        return f"CUDA (GPU {index})"
    return str(device)


def format_bytes_gib(num_bytes: int | None) -> str:
    if num_bytes is None or num_bytes <= 0:
        return "—"
    gib = num_bytes / (1024**3)
    return f"{gib:.1f} Go".replace(".", ",")


def format_hardware_summary(info: HardwareInfo | None = None) -> list[str]:
    """Short lines for a Streamlit expander."""
    snap = info if info is not None else get_hardware_info()
    lines: list[str] = []
    if snap.torch_version:
        lines.append(f"PyTorch : {snap.torch_version}")
    if snap.cuda_available and snap.primary_gpu is not None:
        gpu = snap.primary_gpu
        lines.append(gpu.name)
        lines.append(f"VRAM : {format_bytes_gib(gpu.total_memory_bytes)}")
        if snap.cuda_version:
            lines.append(f"CUDA : {snap.cuda_version}")
        if snap.cudnn_version:
            lines.append(f"cuDNN : {snap.cudnn_version}")
        if snap.device_count > 1:
            lines.append(f"GPU détectés : {snap.device_count}")
    else:
        lines.append("Aucun GPU CUDA disponible")
        if snap.cuda_version:
            lines.append(f"CUDA (build PyTorch) : {snap.cuda_version}")
    return lines


def cuda_oom_user_message(exc: BaseException) -> str | None:
    """Return a user-facing hint when ``exc`` looks like a CUDA OOM error."""
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "out of memory",
        "cuda out of memory",
        "cudnn_status_alloc_failed",
        "hip out of memory",
    )
    if not any(marker in text for marker in markers):
        return None
    return (
        "Mémoire GPU insuffisante (CUDA Out Of Memory). "
        "Essayez de réduire le batch, de diminuer imgsz, "
        "ou d'utiliser un modèle plus petit (ex. YOLO11n)."
    )


def is_cuda_oom_error(exc: BaseException) -> bool:
    return cuda_oom_user_message(exc) is not None
