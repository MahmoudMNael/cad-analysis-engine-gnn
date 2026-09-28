from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from packages.common.geometry import iter_line_segments
from packages.common.spatial_index import SpatialIndex, expand_bounds

FACE_FEATURE_SCHEMA_VERSION = "face_features_v1"
FACE_EDGE_FEATURE_SCHEMA_VERSION = "face_edge_features_v1"

# Canonical GNN-C node schema. Matches §21; centroid/bbox are stored as components.
FACE_NODE_FEATURE_KEYS: tuple[str, ...] = (
    "area",
    "perimeter",
    "compactness",
    "centroid_x",
    "centroid_y",
    "bbox_minx",
    "bbox_miny",
    "bbox_maxx",
    "bbox_maxy",
    "aspect_ratio",
    "number_of_doors",
    "number_of_windows",
    "wall_overlap",
    "text_overlap",
    "adjacent_face_count",
    "adjacent_unbounded",
    "boundary_complexity",
)

# Shared-boundary edge schema for the face-adjacency graph (§22).
FACE_EDGE_FEATURE_KEYS: tuple[str, ...] = (
    "shared_length",
    "shared_length_norm",
    "area_ratio",
    "portal_on_boundary",
)

_PORTAL_BUFFER_M = 0.05


def extract_face_feature_vector(features: Mapping[str, Any]) -> dict[str, float]:
    return {key: _as_float(features.get(key)) for key in FACE_NODE_FEATURE_KEYS}


def vectorize_face_features(features: Mapping[str, Any]) -> list[float]:
    return [extract_face_feature_vector(features)[key] for key in FACE_NODE_FEATURE_KEYS]


def vectorize_edge_features(features: Mapping[str, Any]) -> list[float]:
    return [_as_float(features.get(key)) for key in FACE_EDGE_FEATURE_KEYS]


def _portal_hit(
    inter: BaseGeometry,
    portals: Sequence[BaseGeometry],
    tolerance: float,
    portal_index: SpatialIndex | None,
) -> float:
    radius = max(tolerance, _PORTAL_BUFFER_M)
    if portal_index is not None and not inter.is_empty:
        hit_ids = portal_index.query(expand_bounds(inter, radius))
        portal_iter: Sequence[BaseGeometry] = [portals[int(pid)] for pid in hit_ids]
    else:
        portal_iter = portals
    for portal in portal_iter:
        if portal.is_empty:
            continue
        if portal.distance(inter) <= radius or (
            not inter.is_empty and inter.buffer(_PORTAL_BUFFER_M).intersects(portal)
        ):
            return 1.0
    return 0.0


def _shared_boundary_from_parts(
    boundary_a: BaseGeometry,
    boundary_b: BaseGeometry,
    buffered_b: BaseGeometry,
    *,
    area_a: float,
    area_b: float,
    perimeter_a: float,
    perimeter_b: float,
    portals: Sequence[BaseGeometry],
    tolerance: float,
    portal_index: SpatialIndex | None = None,
) -> dict[str, float] | None:
    if boundary_a.is_empty or boundary_b.is_empty:
        return None
    if boundary_a.distance(boundary_b) > tolerance:
        return None
    inter = boundary_a.intersection(buffered_b)
    shared = float(inter.length) if not inter.is_empty else 0.0
    if shared <= 0.0:
        return None
    min_perim = max(min(perimeter_a, perimeter_b), 1e-9)
    max_area = max(max(area_a, area_b), 1e-9)
    return {
        "shared_length": shared,
        "shared_length_norm": float(min(shared / min_perim, 2.0)),
        "area_ratio": float(min(area_a, area_b) / max_area),
        "portal_on_boundary": _portal_hit(inter, portals, tolerance, portal_index),
    }


def shared_boundary_features(
    geom_a: BaseGeometry,
    geom_b: BaseGeometry,
    *,
    area_a: float,
    area_b: float,
    perimeter_a: float,
    perimeter_b: float,
    portals: Sequence[BaseGeometry] = (),
    tolerance: float = 1e-4,
) -> dict[str, float] | None:
    if geom_a.is_empty or geom_b.is_empty:
        return None
    return _shared_boundary_from_parts(
        geom_a.boundary,
        geom_b.boundary,
        geom_b.boundary.buffer(tolerance),
        area_a=area_a,
        area_b=area_b,
        perimeter_a=perimeter_a,
        perimeter_b=perimeter_b,
        portals=portals,
        tolerance=tolerance,
        portal_index=None,
    )


def build_face_adjacency(
    faces: Sequence[tuple[str, BaseGeometry, float, float]],
    portals: Sequence[BaseGeometry] = (),
    *,
    tolerance: float = 1e-4,
) -> tuple[list[tuple[int, int]], list[list[float]]]:
    """Undirected face graph as bidirectional edges plus shared-boundary attributes."""
    edge_index: list[tuple[int, int]] = []
    edge_attr: list[list[float]] = []
    if not faces:
        return edge_index, edge_attr

    boundaries: list[BaseGeometry] = []
    buffered: list[BaseGeometry] = []
    segment_items: list[tuple[str, BaseGeometry]] = []
    for i, (_, geom, _, _) in enumerate(faces):
        boundary = geom.boundary
        boundaries.append(boundary)
        buffered.append(boundary.buffer(tolerance) if not boundary.is_empty else boundary)
        for a, b in iter_line_segments(boundary):
            segment_items.append((str(i), LineString([a, b])))
    segment_tree = SpatialIndex.build(segment_items)
    portal_index = (
        SpatialIndex.build([(str(i), geom) for i, geom in enumerate(portals) if not geom.is_empty])
        if portals
        else None
    )

    for i, (_, geom_i, area_i, perim_i) in enumerate(faces):
        if geom_i.is_empty:
            continue
        hits: set[int] = set()
        for a, b in iter_line_segments(boundaries[i]):
            probe = expand_bounds(LineString([a, b]), tolerance)
            for jid in segment_tree.query(probe):
                j = int(jid)
                if j > i:
                    hits.add(j)
        for j in sorted(hits):
            _, geom_j, area_j, perim_j = faces[j]
            if geom_j.is_empty:
                continue
            feats = _shared_boundary_from_parts(
                boundaries[i],
                boundaries[j],
                buffered[j],
                area_a=area_i,
                area_b=area_j,
                perimeter_a=perim_i,
                perimeter_b=perim_j,
                portals=portals,
                tolerance=tolerance,
                portal_index=portal_index,
            )
            if feats is None:
                continue
            vec = vectorize_edge_features(feats)
            edge_index.append((i, j))
            edge_attr.append(vec)
            edge_index.append((j, i))
            edge_attr.append(vec)
    return edge_index, edge_attr


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
