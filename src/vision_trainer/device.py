"""Public device / hardware helpers (training + inference)."""

from vision_trainer.training.device import (
    CudaDeviceInfo,
    DeviceChoice,
    DeviceError,
    DeviceOption,
    HardwareInfo,
    cuda_oom_user_message,
    describe_device,
    format_hardware_summary,
    get_available_devices,
    get_hardware_info,
    is_cuda_available,
    is_cuda_oom_error,
    resolve_device,
)

__all__ = [
    "CudaDeviceInfo",
    "DeviceChoice",
    "DeviceError",
    "DeviceOption",
    "HardwareInfo",
    "cuda_oom_user_message",
    "describe_device",
    "format_hardware_summary",
    "get_available_devices",
    "get_hardware_info",
    "is_cuda_available",
    "is_cuda_oom_error",
    "resolve_device",
]
