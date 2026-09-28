"""Spatial indexes for candidate filtering.

Probe envelopes must be a conservative superset of the original predicate
(expanded axis-aligned bounds, or STRtree dwithin). Do not use GEOS buffer()
geometry as the sole probe — buffer approximations can miss true hits.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree


@dataclass
class SpatialIndex:
    ids: tuple[str, ...]
    geoms: tuple[BaseGeometry, ...]
    tree: STRtree

    @classmethod
    def build(cls, items: Sequence[tuple[str, BaseGeometry]]) -> SpatialIndex:
        ids = tuple(i for i, _ in items)
        geoms = tuple(g for _, g in items)
        tree = STRtree(list(geoms)) if geoms else STRtree([])
        return cls(ids=ids, geoms=geoms, tree=tree)

    def query(self, geom: BaseGeometry, predicate: str = "intersects") -> list[str]:
        return [self.ids[i] for i in self.query_indexes(geom, predicate=predicate)]

    def query_indexes(self, geom: BaseGeometry, predicate: str = "intersects") -> list[int]:
        if not self.geoms or geom.is_empty:
            return []
        indexes = self.tree.query(geom, predicate=predicate)
        return [int(i) for i in indexes]


def build_index(items: Sequence[tuple[str, BaseGeometry]]) -> SpatialIndex:
    return SpatialIndex.build(items)


def query_ids(index: SpatialIndex, geom: BaseGeometry, *, predicate: str = "intersects") -> list[str]:
    return index.query(geom, predicate=predicate)


def query_dwithin(index: SpatialIndex, geom: BaseGeometry, distance: float) -> list[str]:
    if not index.geoms or geom.is_empty:
        return []
    indexes = index.tree.query(geom, predicate="dwithin", distance=distance)
    return [index.ids[int(i)] for i in indexes]


def expand_bounds(geom: BaseGeometry, radius: float) -> BaseGeometry:
    """Axis-aligned box around geom.bounds expanded by radius. Never uses buffer()."""
    if geom.is_empty:
        return geom
    minx, miny, maxx, maxy = geom.bounds
    if not all(math.isfinite(v) for v in (minx, miny, maxx, maxy)):
        return geom
    r = max(float(radius), 0.0)
    return box(minx - r, miny - r, maxx + r, maxy + r)
