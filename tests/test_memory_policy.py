from __future__ import annotations

from inline_core.device.memory import MemoryPolicy
from inline_core.device.policy import Profile, Quantization
from inline_core.device.types import Device, DeviceKind

_CUDA = Device(DeviceKind.CUDA, 0)
_CPU = Device(DeviceKind.CPU)


def test_ample_vram_is_gpu_max() -> None:
    policy = MemoryPolicy(_CUDA, vram_gb=24)
    assert policy.profile is Profile.GPU_MAX
    assert policy.placement("denoiser").offload is False
    assert policy.quantization() is Quantization.NONE
    assert policy.attention_slicing() is False


def test_tight_vram_is_lowvram_with_offload() -> None:
    policy = MemoryPolicy(_CUDA, vram_gb=6)
    assert policy.profile is Profile.LOWVRAM
    assert policy.placement("denoiser").offload is True
    assert policy.attention_slicing() is True
    assert policy.vae_tiling() is True
    assert policy.quantization() is Quantization.INT8


def test_cpu_uses_fp32_and_quantizes_on_low_ram() -> None:
    low = MemoryPolicy(_CPU, ram_gb=16)
    assert low.profile is Profile.CPU
    assert low.placement("denoiser").dtype.value == "fp32"
    assert low.placement("denoiser").offload is False
    assert low.quantization() is Quantization.INT8
    assert low.vae_tiling() is True

    ample = MemoryPolicy(_CPU, ram_gb=128)
    assert ample.quantization() is Quantization.NONE


def test_env_profile_override(monkeypatch) -> None:
    monkeypatch.setenv("INLINE_PROFILE", "lowvram")
    assert MemoryPolicy(_CUDA, vram_gb=48).profile is Profile.LOWVRAM
