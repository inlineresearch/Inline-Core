"""Run the engine: `python -m inline_core.server`. Registers models whose deps are installed."""

from __future__ import annotations

import uvicorn

from ..config import data_dir, server_host, server_port
from ..device.memory import MemoryPolicy
from ..graph.cache import InMemoryCache
from ..graph.registry import build_default_registry
from ..runtime.file_store import FileTakeStore
from .app import create_app
from .bootstrap import register_models
from .run_store import SqliteRunStore


def main() -> None:
    policy = MemoryPolicy()
    registry = build_default_registry()
    data = data_dir()
    takes = data / "takes"
    store = FileTakeStore(takes)
    run_store = SqliteRunStore(data / "runs.db")
    registered = register_models(registry, store, policy)
    print(f"Registered models: {registered or 'none (source nodes only)'}")
    app = create_app(
        registry=registry,
        cache=InMemoryCache(),
        policy=policy,
        run_store=run_store,
        takes_dir=str(takes),
    )
    host, port = server_host(), server_port()
    print(f"Serving on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
