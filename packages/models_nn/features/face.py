from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from shapely.geometry.base import BaseGeometry

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


def extract_face_feature_vector(features: Mapping[str, Any]) -> dict[str, float]:
    return {key: _as_float(features.get(key)) for key in FACE_NODE_FEATURE_KEYS}


def vectorize_face_features(features: Mapping[str, Any]) -> list[float]:
    return [extract_face_feature_vector(features)[key] for key in FACE_NODE_FEATURE_KEYS]


def vectorize_edge_features(features: Mapping[str, Any]) -> list[float]:
    return [_as_float(features.get(key)) for key in FACE_EDGE_FEATURE_KEYS]


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
    if geom_a.boundary.distance(geom_b.boundary) > tolerance:
        return None
    inter = geom_a.boundary.intersection(geom_b.boundary.buffer(tolerance))
    shared = float(inter.length) if not inter.is_empty else 0.0
    if shared <= 0.0:
        return None
    min_perim = max(min(perimeter_a, perimeter_b), 1e-9)
    max_area = max(max(area_a, area_b), 1e-9)
    portal_hit = 0.0
    for portal in portals:
        if portal.is_empty:
            continue
        if portal.distance(inter) <= max(tolerance, 0.05) or (
            not inter.is_empty and inter.buffer(0.05).intersects(portal)
        ):
            portal_hit = 1.0
            break
    return {
        "shared_length": shared,
        "shared_length_norm": float(min(shared / min_perim, 2.0)),
        "area_ratio": float(min(area_a, area_b) / max_area),
        "portal_on_boundary": portal_hit,
    }


def build_face_adjacency(
    faces: Sequence[tuple[str, BaseGeometry, float, float]],
    portals: Sequence[BaseGeometry] = (),
    *,
    tolerance: float = 1e-4,
) -> tuple[list[tuple[int, int]], list[list[float]]]:
    """Undirected face graph as bidirectional edges plus shared-boundary attributes."""
    edge_index: list[tuple[int, int]] = []
    edge_attr: list[list[float]] = []
    for i, (_, geom_i, area_i, perim_i) in enumerate(faces):
        for j in range(i + 1, len(faces)):
            _, geom_j, area_j, perim_j = faces[j]
            feats = shared_boundary_features(
                geom_i,
                geom_j,
                area_a=area_i,
                area_b=area_j,
                perimeter_a=perim_i,
                perimeter_b=perim_j,
                portals=portals,
                tolerance=tolerance,
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
