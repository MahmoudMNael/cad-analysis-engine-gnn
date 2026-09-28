from __future__ import annotations

from pathlib import Path

from packages.store.models_registry import PinnedModels, resolve_pinned_models

__all__ = [
    "PinnedModels",
    "checkpoint_object_key",
    "is_gnn_ref",
    "is_heuristic_ref",
    "resolve_pinned_models",
    "uri_to_path",
]


def is_gnn_ref(ref: str) -> bool:
    return str(ref).startswith("gnn:")


def is_heuristic_ref(ref: str) -> bool:
    return str(ref).startswith("heuristic:")


def checkpoint_object_key(model_name: str, version: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in version)
    return f"models/{model_name}/{safe}.pt"


def uri_to_path(uri: str) -> Path:
    if uri.startswith("file://"):
        return Path(uri[7:])
    return Path(uri)
