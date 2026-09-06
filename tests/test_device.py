from __future__ import annotations

import pytest

from vision_trainer.training.device import (
    CudaDeviceInfo,
    DeviceError,
    HardwareInfo,
    cuda_oom_user_message,
    describe_device,
    format_hardware_summary,
    get_available_devices,
    get_hardware_info,
    resolve_device,
)


def test_resolve_device_cpu() -> None:
    assert resolve_device("cpu") == "cpu"
    assert describe_device("cpu") == "CPU"


def test_resolve_device_auto_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vision_trainer.training.device.is_cuda_available",
        lambda: False,
    )
    assert resolve_device("auto") == "cpu"


def test_resolve_device_auto_with_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vision_trainer.training.device.is_cuda_available",
        lambda: True,
    )
    assert resolve_device("auto") == "0"


def test_resolve_device_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vision_trainer.training.device.is_cuda_available",
        lambda: False,
    )
    with pytest.raises(DeviceError, match="CUDA"):
        resolve_device("cuda")


def test_resolve_device_cuda_index(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Cuda:
        @staticmethod
        def device_count() -> int:
            return 2

        @staticmethod
        def is_available() -> bool:
            return True

    class _Torch:
        cuda = _Cuda()

    monkeypatch.setitem(__import__("sys").modules, "torch", _Torch())
    monkeypatch.setattr(
        "vision_trainer.training.device.is_cuda_available",
        lambda: True,
    )
    assert resolve_device("cuda") == "0"
    assert resolve_device("cuda:1") == "1"
    assert resolve_device("1") == "1"
    with pytest.raises(DeviceError, match="GPU 5"):
        resolve_device("cuda:5")


def test_get_available_devices_cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vision_trainer.training.device.get_hardware_info",
        lambda: HardwareInfo(cuda_available=False, device_count=0),
    )
    options = get_available_devices()
    assert [o.choice for o in options] == ["auto", "cpu"]
    assert all("GPU" not in o.label for o in options)


def test_get_available_devices_with_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    info = HardwareInfo(
        cuda_available=True,
        device_count=1,
        cuda_devices=(
            CudaDeviceInfo(index=0, name="NVIDIA GeForce RTX 5070", total_memory_bytes=12 * 1024**3),
        ),
        torch_version="2.5.1",
        cuda_version="12.4",
    )
    monkeypatch.setattr("vision_trainer.training.device.get_hardware_info", lambda: info)
    options = get_available_devices()
    assert options[0].choice == "auto"
    assert options[1].choice == "cpu"
    assert options[2].choice == "cuda"
    assert "RTX 5070" in options[2].label


def test_describe_device_uses_gpu_name(monkeypatch: pytest.MonkeyPatch) -> None:
    info = HardwareInfo(
        cuda_available=True,
        device_count=1,
        cuda_devices=(CudaDeviceInfo(index=0, name="NVIDIA GeForce RTX 5070"),),
    )
    assert describe_device("0", hardware=info) == "GPU — NVIDIA GeForce RTX 5070"
    assert describe_device("cpu", hardware=info) == "CPU"


def test_format_hardware_summary_with_and_without_gpu() -> None:
    cpu_lines = format_hardware_summary(
        HardwareInfo(cuda_available=False, device_count=0, torch_version="2.5.1+cpu")
    )
    assert any("Aucun GPU" in line for line in cpu_lines)
    assert any("PyTorch" in line for line in cpu_lines)

    gpu_lines = format_hardware_summary(
        HardwareInfo(
            cuda_available=True,
            device_count=1,
            cuda_devices=(
                CudaDeviceInfo(index=0, name="NVIDIA GeForce RTX 5070", total_memory_bytes=12 * 1024**3),
            ),
            torch_version="2.5.1",
            cuda_version="12.4",
        )
    )
    assert any("RTX 5070" in line for line in gpu_lines)
    assert any("VRAM" in line for line in gpu_lines)
    assert any("CUDA" in line for line in gpu_lines)


def test_cuda_oom_user_message() -> None:
    msg = cuda_oom_user_message(RuntimeError("CUDA out of memory. Tried to allocate ..."))
    assert msg is not None
    assert "batch" in msg.lower()
    assert cuda_oom_user_message(ValueError("unrelated")) is None


def test_get_hardware_info_handles_cuda_probe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            raise RuntimeError("cuda driver blowup")

    class _Torch:
        __version__ = "2.5.1"
        version = type("V", (), {"cuda": "12.4"})()
        cuda = _Cuda()

    monkeypatch.setitem(__import__("sys").modules, "torch", _Torch())
    info = get_hardware_info()
    assert info.cuda_available is False
    assert info.torch_version == "2.5.1"
