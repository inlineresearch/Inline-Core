"""The device and memory policy interface. Components never self-assign a device; they ask here."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from .types import Device, DType


class Profile(str, Enum):
    GPU_MAX = "gpu-max"
    LOWVRAM = "lowvram"
    CPU = "cpu"


class AttentionBackend(str, Enum):
    FLASH = "flash"
    XFORMERS = "xformers"
    SDPA = "sdpa"


@dataclass(frozen=True)
class Placement:
    """Where and how a component runs. Chosen by the policy, never by the component."""

    device: Device
    dtype: DType
    offload: bool = False


class DevicePolicy(ABC):
    """Owns dtype, device, offload, attention backend, and tiling for a worker."""

    @property
    @abstractmethod
    def profile(self) -> Profile: ...

    @abstractmethod
    def placement(self, role: str) -> Placement:
        """Placement for a component role: text_encoder, denoiser, vae, and so on."""

    @abstractmethod
    def attention_backend(self) -> AttentionBackend: ...

    @abstractmethod
    def vae_tiling(self) -> bool:
        """Whether to tile VAE decode to cap peak memory."""
