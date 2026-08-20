"""Spatial indexes built immediately after primitive extraction."""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from packages.ingest.models import Primitive
from packages.common.geometry import iter_line_segments


@dataclass
class SpatialIndex:
    ids: tuple[str, ...]
    geoms: tuple[BaseGeometry, ...]
    tree: STRtree

    @classmethod
    def build(cls, items: list[tuple[str, BaseGeometry]]) -> SpatialIndex:
        ids = tuple(i for i, _ in items)
        geoms = tuple(g for _, g in items)
        tree = STRtree(list(geoms)) if geoms else STRtree([])
        return cls(ids=ids, geoms=geoms, tree=tree)

    def query(self, geom: BaseGeometry, predicate: str = "intersects") -> list[str]:
        if not self.geoms:
            return []
        indexes = self.tree.query(geom, predicate=predicate)
        return [self.ids[int(i)] for i in indexes]


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
