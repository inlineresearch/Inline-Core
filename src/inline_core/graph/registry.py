"""The node registry: descriptors served at /v1/models plus the runner behind each type."""

from __future__ import annotations

import hashlib
import json

from ..errors import UnknownNodeType
from .descriptor import NodeDescriptor
from .runners import IMAGE_INPUT, TEXT_INPUT, ImageInputRunner, NodeRunner, TextInputRunner


class Registry:
    def __init__(self) -> None:
        self._descriptors: dict[str, NodeDescriptor] = {}
        self._runners: dict[str, NodeRunner] = {}

    def register(self, descriptor: NodeDescriptor, runner: NodeRunner) -> None:
        self._descriptors[descriptor.type] = descriptor
        self._runners[descriptor.type] = runner

    def get(self, node_type: str) -> NodeDescriptor:
        descriptor = self._descriptors.get(node_type)
        if descriptor is None:
            raise UnknownNodeType(f"Unknown node type {node_type!r}.")
        return descriptor

    def has(self, node_type: str) -> bool:
        return node_type in self._descriptors

    def runner(self, node_type: str) -> NodeRunner:
        runner = self._runners.get(node_type)
        if runner is None:
            raise UnknownNodeType(f"No runner registered for {node_type!r}.")
        return runner

    def descriptors(self) -> list[NodeDescriptor]:
        return list(self._descriptors.values())

    def version(self) -> str:
        # TODO(phase1): fold resolved dynamic options into this so installing a model bumps it.
        payload = json.dumps(sorted(self._descriptors), separators=(",", ":"))
        return f"r_{hashlib.sha256(payload.encode()).hexdigest()[:8]}"


def build_default_registry() -> Registry:
    """A registry with the built-in source nodes. Models register onto it as they load."""
    registry = Registry()
    registry.register(TEXT_INPUT, TextInputRunner())
    registry.register(IMAGE_INPUT, ImageInputRunner())
    return registry
