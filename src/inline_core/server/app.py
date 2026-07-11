"""The FastAPI app: the /v1 routes from docs/contract.md over the run manager and registry."""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response, WebSocket
from fastapi.responses import FileResponse, JSONResponse

from ..config import models_dir
from ..device.auto import AutoDevicePolicy
from ..device.policy import DevicePolicy
from ..errors import GraphValidationError, UnknownNodeType
from ..graph.cache import InMemoryCache, NodeCache
from ..graph.registry import Registry, build_default_registry
from ..graph.schema import SCHEMA_VERSION, parse_graph
from ..models.catalog import ModelCatalog
from .assets import AssetStore
from .manager import RunConflict, RunManager
from .run_store import RunStore
from .serialize import descriptor_json, event_json, run_json, take_json


def _within(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _error(code: str, message: str, status: int, node_id: str | None = None) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if node_id is not None:
        error["nodeId"] = node_id
    return JSONResponse({"error": error}, status_code=status)


def _version(registry: Registry, catalog: ModelCatalog) -> str:
    """Registry version = node types + the scanned model files, so dropping a file bumps it."""
    payload = json.dumps(
        {"types": sorted(d.type for d in registry.descriptors()), "models": catalog.fingerprint()}
    )
    return f"r_{hashlib.sha256(payload.encode()).hexdigest()[:8]}"


def create_app(
    registry: Registry | None = None,
    cache: NodeCache | None = None,
    policy: DevicePolicy | None = None,
    asset_dir: str | None = None,
    models_root: str | None = None,
    run_store: RunStore | None = None,
    takes_dir: str | None = None,
) -> FastAPI:
    registry = registry or build_default_registry()
    cache = cache or InMemoryCache()
    policy = policy or AutoDevicePolicy()
    assets = AssetStore(Path(asset_dir or "./.inline-assets"))
    catalog = ModelCatalog(Path(models_root) if models_root else models_dir())
    takes_root = Path(takes_dir or "./.inline-takes")
    manager = RunManager(registry, cache, policy, store=run_store)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ANN202
        manager.bind_loop(asyncio.get_running_loop())
        catalog.ensure_dirs()
        catalog.scan()
        yield
        manager.shutdown()

    app = FastAPI(title="Inline Core", version="0.0.0", lifespan=lifespan)

    @app.get("/v1/health")
    async def health() -> dict[str, Any]:
        placement = policy.placement("denoiser")
        return {
            "ok": True,
            "apiVersion": "v1",
            "schemaVersions": {"min": SCHEMA_VERSION, "max": SCHEMA_VERSION},
            "registryVersion": _version(registry, catalog),
            "device": {
                "kind": placement.device.kind.value,
                "profile": policy.profile.value,
                "vramBudgetMb": None,
            },
        }

    @app.get("/v1/models")
    async def list_models(request: Request) -> Response:
        version = _version(registry, catalog)
        etag = f'"{version}"'
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304)
        body = {
            "registryVersion": version,
            "models": [descriptor_json(d, catalog) for d in registry.descriptors()],
        }
        return JSONResponse(body, headers={"ETag": etag})

    @app.get("/v1/models/{model_type:path}")
    async def get_model(model_type: str) -> Response:
        try:
            return JSONResponse(descriptor_json(registry.get(model_type), catalog))
        except UnknownNodeType as error:
            return _error("not_found", str(error), 404)

    @app.post("/v1/runs")
    async def submit_run(request: Request) -> Response:
        body = await request.json()
        target = body.get("target")
        if not isinstance(target, str):
            return _error("invalid_request", "'target' is required.", 422)
        try:
            graph = parse_graph(body.get("graph"))
            record, created = manager.submit(graph, target, body.get("clientRunId"))
        except GraphValidationError as error:
            return _error("invalid_graph", str(error), 422, node_id=error.node_id)
        except RunConflict as error:
            return _error("conflict", str(error), 409)
        return JSONResponse(
            {"runId": record.state.run_id, "status": record.state.status.value},
            status_code=201 if created else 200,
        )

    @app.get("/v1/runs/{run_id}")
    async def get_run(run_id: str) -> Response:
        record = manager.get(run_id)
        if record is None:
            return _error("not_found", f"No run {run_id!r}.", 404)
        return JSONResponse(run_json(record.state))

    @app.delete("/v1/runs/{run_id}")
    async def cancel_run(run_id: str) -> Response:
        if not manager.cancel(run_id):
            return _error("not_found", f"No run {run_id!r}.", 404)
        return JSONResponse({"runId": run_id, "status": "cancelled"})

    @app.post("/v1/assets")
    async def upload_asset(request: Request) -> Response:
        data = await request.body()
        stored = assets.put(data, request.headers.get("content-type"))
        return JSONResponse({"id": stored.id, "kind": stored.kind.value, "bytes": stored.size})

    @app.get("/v1/takes/{take_id}")
    async def get_take(take_id: str) -> Response:
        take = manager.find_take(take_id)
        if take is None:
            return _error("not_found", f"No take {take_id!r}.", 404)
        return JSONResponse(take_json(take))

    @app.get("/v1/takes/{take_id}/bytes")
    async def get_take_bytes(take_id: str) -> Response:
        take = manager.find_take(take_id)
        if take is None:
            return _error("not_found", f"No take {take_id!r}.", 404)
        path = Path(take.uri)
        if not _within(takes_root, path) or not path.is_file():
            return _error("not_found", "Take bytes are not available.", 404)
        return FileResponse(path)

    @app.websocket("/v1/runs/{run_id}/events")
    async def run_events(websocket: WebSocket, run_id: str) -> None:
        record = manager.get(run_id)
        if record is None:
            await websocket.close(code=4404)
            return
        await websocket.accept()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        record.subscribers.add(queue)
        try:
            await websocket.send_json(
                {"type": "snapshot", "runId": run_id, "state": run_json(record.state)}
            )
            if record.done:
                return
            while True:
                event = await queue.get()
                if event is None:
                    break
                await websocket.send_json(event_json(event))
        finally:
            record.subscribers.discard(queue)

    return app
