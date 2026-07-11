"""The graph/GPU boundary. The graph never runs the denoise loop inline; it submits a SampleJob.

Phase 1 runs one job at a time behind this seam. Phase 5 adds cross-request batching (grouping by
model family, resolution, and adapter bucket) without changing the interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..components.conditioning import Conditioning, Latents
from ..components.interfaces import Denoiser, Sampler, Scheduler, StepCallback

if TYPE_CHECKING:
    from ..runtime.context import ExecutionContext


@dataclass
class SampleJob:
    """One request to sample. Compatible jobs are grouped and stepped together."""

    denoiser: Denoiser
    scheduler: Scheduler
    sampler: Sampler
    latents: Latents
    conditioning: Conditioning
    steps: int
    on_step: StepCallback | None = None


class BatchedSampler(ABC):
    @abstractmethod
    def submit(self, job: SampleJob, ctx: ExecutionContext) -> Latents: ...


class InlineBatchedSampler(BatchedSampler):
    """Phase 1: no batching. Run the job through its sampler directly."""

    def submit(self, job: SampleJob, ctx: ExecutionContext) -> Latents:
        return job.sampler.sample(
            job.denoiser,
            job.scheduler,
            job.latents,
            job.conditioning,
            steps=job.steps,
            ctx=ctx,
            on_step=job.on_step,
        )
