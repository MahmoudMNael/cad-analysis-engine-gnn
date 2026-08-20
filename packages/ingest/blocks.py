"""Flatten INSERT/block transforms while retaining the original block name."""

from __future__ import annotations

from collections.abc import Iterator

from ezdxf.entities import DXFEntity, Insert


def flatten_insert(entity: Insert, *, depth: int = 0, max_depth: int = 8) -> Iterator[tuple[DXFEntity, str]]:
    block_name = str(entity.dxf.name)
    if depth >= max_depth:
        return
    try:
        virtual = list(entity.virtual_entities())
    except Exception:
        return
    for child in virtual:
        if isinstance(child, Insert):
            yield from flatten_insert(child, depth=depth + 1, max_depth=max_depth)
        else:
            yield child, block_name
