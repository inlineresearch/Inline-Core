"""Z-Image Turbo runner: prompt (+ optional image) -> one rendered take.

A single generation node, ``alibaba/z-image-turbo``, backed by diffusers' ``ZImagePipeline``
(text-to-image) and ``ZImageImg2ImgPipeline`` (when an image is wired in). The heavy pipeline is
built once and cached across runs; only the descriptor is cheap. Placement (device, dtype, offload,
tiling) comes from the DevicePolicy — the runner never picks a device itself. Decoded images are
handed to the TakeStore, which owns bytes/hash/uri.

torch + diffusers are imported at module top on purpose: an absent ``zimage`` extra makes this
import raise, and ``server.bootstrap`` skips the model (best-effort) so the engine still boots.
"""

from __future__ import annotations

import os
import random
from threading import Lock
from typing import Any

import torch
from diffusers import ZImageImg2ImgPipeline, ZImagePipeline

from ...config import models_dir
from ...device.policy import DevicePolicy, Placement, Profile
from ...device.types import DType
from ...errors import CancelledError, ComponentError
from ...graph.descriptor import NodeDescriptor, ParamField, Port, Widget
from ...graph.runners import NodeResult, NodeRunner
from ...graph.schema import Node, PortKind
from ...media import MediaKind
from ...runtime.context import ExecutionContext
from ...runtime.progress import Phase, ProgressEvent
from ...runtime.store import TakeStore
from ...takes import AssetRef

# The default weights, overridable with INLINE_ZIMAGE_MODEL (a HF repo id or a local diffusers dir).
# Tongyi-MAI is Alibaba Tongyi's official Z-Image repo; a full folder dropped under the models root
# (models/diffusion_models/<name>/) is preferred over a network download.
_DEFAULT_MODEL = "Tongyi-MAI/Z-Image-Turbo"
_LOCAL_NAMES = ("Z-Image-Turbo", "z-image-turbo", "Z-Image", "z-image")

_SEED_MAX = 2**31 - 1


ZIMAGE = NodeDescriptor(
    type="alibaba/z-image-turbo",
    title="Z-Image Turbo",
    category="Generate",
    icon="wand",
    output_kind=MediaKind.IMAGE,
    inputs=(
        Port("prompt", "Prompt", PortKind.TEXT, required=True),
        # Optional: wire an image to run img2img instead of text-to-image.
        Port("image", "Image (img2img)", PortKind.IMAGE, required=False),
    ),
    outputs=(Port("image", "Image", PortKind.IMAGE),),
    params=(
        ParamField("negative_prompt", "Negative prompt", Widget.TEXTAREA, ""),
        ParamField("width", "Width", Widget.NUMBER, 1024, min=256, max=2048, step=64),
        ParamField("height", "Height", Widget.NUMBER, 1024, min=256, max=2048, step=64),
        # Z-Image-Turbo is distilled: ~8 steps, CFG off (guidance 0). See the model card.
        ParamField("steps", "Steps", Widget.NUMBER, 8, min=1, max=100, step=1),
        ParamField("guidance", "Guidance (CFG)", Widget.NUMBER, 0.0, min=0.0, max=20.0, step=0.5),
        # img2img only: how far to move from the input image (0 = keep, 1 = ignore).
        ParamField(
            "strength", "Denoise strength", Widget.NUMBER, 0.6, min=0.0, max=1.0, step=0.05,
            advanced=True,
        ),
        ParamField("seed", "Seed (-1 = random)", Widget.SEED, -1),
    ),
)


def register_zimage(registry: Any, store: TakeStore, policy: DevicePolicy) -> None:
    """Register the Z-Image node and its runner. Called best-effort by server.bootstrap."""
    registry.register(ZIMAGE, ZImageRunner(store, policy))


class ZImageRunner(NodeRunner):
    produces_takes = True

    def __init__(self, store: TakeStore, policy: DevicePolicy) -> None:
        self._store = store
        self._policy = policy

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        prompt = _first_str(inputs.get("prompt"))
        if not prompt:
            raise ComponentError("Z-Image needs a prompt.")
        params = {**ZIMAGE.defaults(), **node.params}
        width, height = int(params["width"]), int(params["height"])
        steps = max(1, int(params["steps"]))
        guidance = float(params["guidance"])
        negative = str(params.get("negative_prompt") or "").strip() or None
        seed = _resolve_seed(params.get("seed"))
        image_ref = _first(inputs.get("image"))
        img2img = image_ref is not None

        ctx.emitter.emit(
            _progress(ctx, node, Phase.LOADING, 0.0, status="Loading Z-Image")
        )
        pipe = _load_pipeline(self._policy, img2img=img2img)

        placement = self._policy.placement("denoiser")
        gen_device = "cpu" if (placement.offload or self._policy.profile is Profile.CPU) else str(
            placement.device
        )
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        def on_step_end(_pipe: Any, step: int, _t: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
            if ctx.cancel.cancelled:
                raise CancelledError("Run cancelled.")
            done = step + 1
            ctx.emitter.emit(
                _progress(ctx, node, Phase.SAMPLE, done / steps, step=done, step_count=steps)
            )
            return kwargs

        call: dict[str, Any] = dict(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            output_type="pil",
            callback_on_step_end=on_step_end,
        )
        if negative is not None:
            call["negative_prompt"] = negative
        if img2img:
            call["image"] = _load_image(image_ref)
            call["strength"] = float(params.get("strength", 0.6))

        image = pipe(**call).images[0]

        ctx.emitter.emit(_progress(ctx, node, Phase.SAVE, 1.0))
        take = self._store.save(
            ctx.run_id,
            node.id,
            image,
            {
                "model": _model_source(),
                "prompt": prompt,
                "negative_prompt": negative or "",
                "width": width,
                "height": height,
                "steps": steps,
                "guidance": guidance,
                "seed": seed,
                **({"strength": call["strength"]} if img2img else {}),
            },
        )
        return NodeResult(outputs={"image": take}, takes=[take])


# --- pipeline cache -----------------------------------------------------------------------------

# Keyed by (model source, img2img). Built once; diffusers pipelines are not thread-safe, but the run
# manager executes one run at a time (workers=1). The lock guards concurrent first-time builds.
_PIPELINES: dict[tuple[str, bool], Any] = {}
_LOCK = Lock()


def _load_pipeline(policy: DevicePolicy, *, img2img: bool) -> Any:
    source = _model_source()
    key = (source, img2img)
    with _LOCK:
        cached = _PIPELINES.get(key)
        if cached is not None:
            return cached
        # An img2img pipe can reuse the base pipe's already-placed weights (no second load).
        base = _PIPELINES.get((source, False))
        if img2img and base is not None:
            pipe = ZImageImg2ImgPipeline.from_pipe(base)
        else:
            cls = ZImageImg2ImgPipeline if img2img else ZImagePipeline
            dtype = _torch_dtype(policy.placement("denoiser"))
            pipe = cls.from_pretrained(source, torch_dtype=dtype)
            _configure(pipe, policy)
        _PIPELINES[key] = pipe
        return pipe


def _configure(pipe: Any, policy: DevicePolicy) -> None:
    placement = policy.placement("denoiser")
    if placement.offload:
        pipe.enable_model_cpu_offload()  # lowvram: stream modules on/off the GPU
    else:
        pipe.to(str(placement.device))
    if policy.attention_slicing():
        _try(pipe.enable_attention_slicing)
    if policy.vae_tiling():
        _try(pipe.enable_vae_tiling)
    _try(pipe.set_progress_bar_config, disable=True)


def _model_source() -> str:
    env = os.environ.get("INLINE_ZIMAGE_MODEL", "").strip()
    if env:
        return env
    root = models_dir() / "diffusion_models"
    for name in _LOCAL_NAMES:
        candidate = root / name
        if (candidate / "model_index.json").is_file():
            return str(candidate)
    return _DEFAULT_MODEL


def _torch_dtype(placement: Placement) -> Any:
    return {
        DType.FP16: torch.float16,
        DType.BF16: torch.bfloat16,
        DType.FP32: torch.float32,
    }.get(placement.dtype, torch.bfloat16)


# --- small helpers ------------------------------------------------------------------------------


def _resolve_seed(raw: Any) -> int:
    """A fixed non-negative seed passes through; -1 (or anything invalid) becomes a fresh random."""
    try:
        seed = int(raw)
    except (TypeError, ValueError):
        seed = -1
    return seed if seed >= 0 else random.randint(0, _SEED_MAX)


def _load_image(ref: Any) -> Any:
    from PIL import Image

    if isinstance(ref, AssetRef) and ref.ref == "path" and ref.path:
        return Image.open(ref.path).convert("RGB")
    raise ComponentError("Z-Image img2img needs a readable image path input.")


def _first(values: list[Any] | None) -> Any:
    return values[0] if values else None


def _first_str(values: list[Any] | None) -> str:
    value = _first(values)
    return str(value).strip() if value is not None else ""


def _progress(
    ctx: ExecutionContext,
    node: Node,
    phase: Phase,
    fraction: float,
    *,
    step: int | None = None,
    step_count: int | None = None,
    status: str = "",
) -> ProgressEvent:
    return ProgressEvent(
        run_id=ctx.run_id,
        node_id=node.id,
        phase=phase,
        fraction=fraction,
        step=step,
        step_count=step_count,
        status=status,
    )


def _try(fn: Any, *args: Any, **kwargs: Any) -> None:
    """Best-effort optional pipeline tweak; skip if this diffusers build lacks it."""
    try:
        fn(*args, **kwargs)
    except (AttributeError, ValueError, NotImplementedError):
        pass
