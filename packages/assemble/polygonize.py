"""Deterministic Shapely polygonization. All faces are retained."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize_full, unary_union

from packages.common.configuration import Settings
from packages.common.geometry import (
    canonicalize_polygon_coords,
    compactness,
    finite_geometry,
    repair_polygon,
)
from packages.common.hashing import stable_hash
from packages.graphkit.models import PlanGraph


@dataclass(frozen=True)
class Face:
    id: str
    geometry: Polygon
    area: float
    perimeter: float
    repaired: bool
    valid: bool
    discarded_reason: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FaceSet:
    faces: tuple[Face, ...]
    unbounded_edge_ids: tuple[str, ...] = ()


def _face_id(polygon: Polygon, precision: int) -> str:
    coords = canonicalize_polygon_coords(list(polygon.exterior.coords), precision)
    return stable_hash("face", coords, {"holes": len(polygon.interiors)})


def polygonize_graph(graph: PlanGraph, settings: Settings) -> tuple[FaceSet, list[dict[str, Any]]]:
    lines = [e.geometry for e in graph.edges if not e.geometry.is_empty]
    notes: list[dict[str, Any]] = []
    if not lines:
        return FaceSet(faces=()), notes
    noded = unary_union(lines)
    polygons, dangles, cuts, invalids = polygonize_full(noded)
    if not dangles.is_empty:
        notes.append({"code": "dangles", "detail": dangles.geom_type})
    if not cuts.is_empty:
        notes.append({"code": "cuts", "detail": cuts.geom_type})
    if not invalids.is_empty:
        notes.append({"code": "invalids", "detail": invalids.geom_type})

    geoms = list(polygons.geoms) if polygons.geom_type == "GeometryCollection" else [polygons]
    edge_uses: dict[str, int] = {e.id: 0 for e in graph.edges}
    faces: list[Face] = []
    for geom in geoms:
        if geom.is_empty or geom.geom_type != "Polygon":
            continue
        repaired, was_repaired = repair_polygon(geom)
        if repaired.geom_type == "MultiPolygon":
            parts = list(repaired.geoms)
        else:
            parts = [repaired]
        for part in parts:
            if part.geom_type != "Polygon" or part.is_empty:
                continue
            valid = bool(part.is_valid and finite_geometry(part) and part.area > 0)
            reason = None
            if not finite_geometry(part):
                reason = "non_finite"
                valid = False
            elif part.area < settings.face_min_area_m2:
                reason = "min_area"
            elif abs(part.bounds[0]) > settings.max_bbox_extent_m:
                reason = "unreasonable_bounds"
                valid = False
            face = Face(
                id=_face_id(part, settings.coord_precision),
                geometry=part,
                area=float(part.area),
                perimeter=float(part.length),
                repaired=was_repaired,
                valid=valid,
                discarded_reason=reason if reason or not valid else None,
                metadata={"compactness": compactness(float(part.area), float(part.length))},
            )
            faces.append(face)
            for edge in graph.edges:
                if edge.geometry.intersects(part.boundary.buffer(settings.intersection_tolerance_m * 5)):
                    if edge.geometry.distance(part) <= settings.intersection_tolerance_m * 5:
                        edge_uses[edge.id] = edge_uses.get(edge.id, 0) + 1
    unique: dict[str, Face] = {}
    for face in faces:
        unique[face.id] = face
    unbounded = tuple(eid for eid, count in edge_uses.items() if count <= 1)
    return FaceSet(faces=tuple(unique.values()), unbounded_edge_ids=unbounded), notes
