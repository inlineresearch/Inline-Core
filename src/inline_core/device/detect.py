"""Best-available device detection. Torch is imported lazily so the core imports without it."""

from __future__ import annotations

from .types import Device, DeviceKind


def available_device() -> Device:
    """cuda, else mps, else cpu."""
    try:
        import torch
    except ModuleNotFoundError:
        return Device(DeviceKind.CPU)
    if torch.cuda.is_available():
        return Device(DeviceKind.CUDA, 0)
    if torch.backends.mps.is_available():
        return Device(DeviceKind.MPS)
    return Device(DeviceKind.CPU)
