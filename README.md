# Inline Core

The generation engine behind Inline. It takes a typed node graph (JSON) and returns immutable renders
("takes"), running image and video models across GPUs, low-VRAM machines, and CPU-only boxes on macOS,
Windows, and Linux. It is the render backend that replaces ComfyUI for Inline.

First model: Z-Image (Alibaba Tongyi), a 6B rectified-flow diffusion transformer.

> Status: early, and running end to end against a stub engine. In place and tested: the graph engine,
> the typed `/v1` HTTP + websocket API (durable runs, streamed progress, coalescing), the model-dir
> scan, the device + memory policy (profiles, dtype, offload, int8), the low-level primitive node
> vocabulary, and a ComfyUI workflow importer. The Z-Image loader is written and validates on a GPU.
> Cross-request batching and out-of-process custom nodes are designed as seams but not yet built.

## How it differs from ComfyUI's architecture

ComfyUI is a great canvas but a fragile engine. Inline Core keeps the open node-graph model and
rebuilds the engine underneath it.

| | ComfyUI | Inline Core |
| --- | --- | --- |
| Graph vs GPU | runs the denoise loop inline, one request at a time | graph orchestration (cheap, per request) is separate from a batched sampler that groups compatible jobs across requests |
| Schema | positional `widgets_values`, validated at runtime (dies mid-graph) | typed graph, named params, edges type-checked before the run (rejected at submit) |
| Devices | some nodes pin to CPU on a GPU box; Z-Image will not run on CPU | one device/memory policy owns dtype, placement, offload, and attention; no node hardcodes a device |
| Custom nodes | all load into one interpreter and env, so any node can break the core | run out of process, each pack in its own venv, behind a semver SDK |
| Interface | a web UI driven by graph JSON over a socket; run state is ephemeral | a headless `/v1` HTTP + websocket API; runs are durable and survive a restart |
| Outputs | files you overwrite | immutable takes; regenerating adds a take, never overwrites |
| Models | `models/` dir, dropdowns from a scan | same layout (bring your own, no downloads); a typed catalog feeds versioned node descriptors the UI renders generically |

The two boundaries that matter most: graph orchestration is decoupled from GPU batching (graphs are
the unit of caching, the sampler is the unit of batching), and the device policy is the single owner
of placement, so the same graph runs on a 4090, a 6 GB laptop, or pure CPU.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```
uv venv
uv pip install -e ".[server]"     # engine + HTTP/websocket API
uv pip install -e ".[zimage]"     # + torch, diffusers, transformers (for real generation)
```

## Models

Bring your own weights; nothing is downloaded. Drop files into the models dir (default `./models`,
override with `INLINE_MODELS_DIR`), ComfyUI-style, by category:

```
models/
  diffusion_models/   z_image_turbo_bf16.safetensors
  vae/                ae.safetensors
  text_encoders/      qwen3-4b/     (a folder: config + tokenizer + weights)
  loras/  controlnet/  checkpoints/  ...
```

The engine scans this on start; a node's model pickers list what is present.

## Nodes and workflows

The canvas (Storyline) wires **low-level primitive nodes** by typed edges. `/v1/models` serves each
node's descriptor (ports, params, file pickers), so the UI renders any node generically and adding a
node type needs no UI release.

- Loaders: `load/diffusion-model`, `load/vae`, `load/text-encoder`
- Conditioning: `encode/text`
- Latent and sampling: `latent/empty`, `sample`
- VAE: `vae/decode`, `vae/encode`

Engine handles (`model`, `vae`, `text-encoder`, `conditioning`, `latent`) are typed sockets passed
between nodes; only media outputs (`vae/decode`) become Frames with take history, the rest are
ephemeral plumbing. A best-effort ComfyUI importer maps existing workflows onto these nodes.

## Run

```
python -m inline_core.server        # serves http://127.0.0.1:8848
```

Working data (the run database and takes) lives in `INLINE_DATA_DIR` (default `./.inline`).

## API (v1)

- `POST /v1/runs {graph, target}` returns `{runId}` (validated up front; 422 on a bad graph)
- `GET /v1/runs/{id}` returns run state (durable; survives a restart)
- `GET /v1/runs/{id}/events` (websocket): a snapshot, then `progress` / `node_done` / `run_done`
- `DELETE /v1/runs/{id}` cancels
- `GET /v1/models` returns node descriptors + `registryVersion` (ETag-aware)
- `GET /v1/takes/{id}` and `/v1/takes/{id}/bytes`
- `POST /v1/assets` (content-addressed upload) and `GET /v1/health`

## Development

```
ruff check .        # lint
uv run pytest -q    # tests (no GPU needed; model code is import-guarded)
```
