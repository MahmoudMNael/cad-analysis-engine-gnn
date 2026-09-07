"""Spatial indexes built immediately after primitive extraction."""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from packages.common.geometry import iter_line_segments
from packages.common.spatial_index import SpatialIndex, build_index, expand_bounds, query_ids
from packages.ingest.models import Primitive

__all__ = [
    "PrimitiveIndexes",
    "SpatialIndex",
    "build_index",
    "build_primitive_indexes",
    "expand_bounds",
    "query_ids",
]


@dataclass(frozen=True)
class PrimitiveIndexes:
    bbox: SpatialIndex
    endpoint: SpatialIndex
    text: SpatialIndex
    block: SpatialIndex


def build_primitive_indexes(primitives: tuple[Primitive, ...]) -> PrimitiveIndexes:
    bbox_items = [(p.id, p.geometry) for p in primitives]
    endpoints: list[tuple[str, BaseGeometry]] = []
    for prim in primitives:
        for a, b in iter_line_segments(prim.geometry):
            endpoints.append((prim.id, Point(a)))
            endpoints.append((prim.id, Point(b)))
        if prim.geometry.geom_type == "Point":
            endpoints.append((prim.id, prim.geometry))
    texts = [(p.id, p.geometry) for p in primitives if p.kind.value == "text"]
    blocks = [(p.id, p.geometry) for p in primitives if p.block_name]
    return PrimitiveIndexes(
        bbox=SpatialIndex.build(bbox_items),
        endpoint=SpatialIndex.build(endpoints),
        text=SpatialIndex.build(texts),
        block=SpatialIndex.build(blocks),
    )
