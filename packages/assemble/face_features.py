from __future__ import annotations

from dataclasses import dataclass

from packages.assemble.polygonize import Face, FaceSet
from packages.common.geometry import aspect_ratio, compactness
from packages.common.spatial_index import SpatialIndex, expand_bounds
from packages.graphkit.models import GraphEdge, PlanGraph, Portal
from packages.ingest.models import Primitive
from packages.models_nn.features.face import FACE_FEATURE_SCHEMA_VERSION


@dataclass(frozen=True)
class FaceFeatureIndex:
    portals: SpatialIndex
    walls: SpatialIndex
    texts: SpatialIndex
    faces: SpatialIndex
    edges: SpatialIndex
    face_by_id: dict[str, Face]
    edge_by_id: dict[str, GraphEdge]
    portal_by_id: dict[str, Portal]
    wall_by_id: dict[str, Primitive]
    text_by_id: dict[str, Primitive]


def build_face_feature_index(
    faces: FaceSet,
    graph: PlanGraph,
    texts: tuple[Primitive, ...],
    walls: tuple[Primitive, ...],
) -> FaceFeatureIndex:
    return FaceFeatureIndex(
        portals=SpatialIndex.build([(p.id, p.geometry) for p in graph.portals]),
        walls=SpatialIndex.build([(w.id, w.geometry) for w in walls]),
        texts=SpatialIndex.build([(t.id, t.geometry) for t in texts]),
        faces=SpatialIndex.build([(f.id, f.geometry.boundary) for f in faces.faces]),
        edges=SpatialIndex.build([(e.id, e.geometry) for e in graph.edges]),
        face_by_id={f.id: f for f in faces.faces},
        edge_by_id=graph.edge_map(),
        portal_by_id={p.id: p for p in graph.portals},
        wall_by_id={w.id: w for w in walls},
        text_by_id={t.id: t for t in texts},
    )


def extract_face_features(
    face: Face,
    faces: FaceSet,
    graph: PlanGraph,
    portals: tuple[Portal, ...],
    texts: tuple[Primitive, ...],
    walls: tuple[Primitive, ...],
    index: FaceFeatureIndex | None = None,
) -> dict[str, float]:
    geom = face.geometry
    minx, miny, maxx, maxy = geom.bounds
    centroid = geom.centroid
    geom_buf = geom.buffer(0.05)
    boundary = geom.boundary
    boundary_buf = boundary.buffer(0.05)

    if index is None:
        portal_iter: tuple[Portal, ...] | list[Portal] = portals
        wall_iter: tuple[Primitive, ...] | list[Primitive] = walls
        text_iter: tuple[Primitive, ...] | list[Primitive] = texts
        other_iter = [other for other in faces.faces if other.id != face.id]
        edge_iter = graph.edges
        edge_map = graph.edge_map()
    else:
        portal_iter = [
            index.portal_by_id[pid]
            for pid in index.portals.query(expand_bounds(geom, 0.05))
            if pid in index.portal_by_id
        ]
        wall_iter = [
            index.wall_by_id[wid]
            for wid in index.walls.query(expand_bounds(boundary, 0.05))
            if wid in index.wall_by_id
        ]
        text_ids = set(index.texts.query(expand_bounds(geom, 0.0))) | set(index.texts.query(centroid))
        text_iter = [index.text_by_id[tid] for tid in text_ids if tid in index.text_by_id]
        other_iter = [
            index.face_by_id[oid]
            for oid in index.faces.query(expand_bounds(boundary, 1e-4))
            if oid != face.id and oid in index.face_by_id
        ]
        edge_iter = [
            index.edge_by_id[eid]
            for eid in index.edges.query(expand_bounds(boundary, 1e-3))
            if eid in index.edge_by_id
        ]
        edge_map = index.edge_by_id

    doors = 0
    windows = 0
    for portal in portal_iter:
        if portal.geometry.distance(geom) <= 1e-4 or geom_buf.intersects(portal.geometry):
            if portal.kind.value == "door":
                doors += 1
            elif portal.kind.value == "window":
                windows += 1
    wall_overlap = 0.0
    for wall in wall_iter:
        inter = wall.geometry.intersection(boundary_buf)
        wall_overlap += float(inter.length) if not inter.is_empty else 0.0
    text_overlap = 0.0
    for text in text_iter:
        if geom.intersects(text.geometry) or geom.contains(text.geometry.centroid):
            text_overlap += 1.0
    adjacent = 0
    adjacent_unbounded = 0
    for other in other_iter:
        if face.geometry.boundary.intersects(other.geometry.boundary.buffer(1e-4)):
            adjacent += 1
    shared = 0.0
    for edge in edge_iter:
        if boundary.distance(edge.geometry) <= 1e-3:
            shared += edge.geometry.length
    if shared < face.perimeter * 0.95:
        adjacent_unbounded = 1
    outer_hits = 0
    for eid in faces.unbounded_edge_ids:
        edge = edge_map.get(eid)
        if edge is None:
            continue
        if boundary.distance(edge.geometry) <= 1e-3:
            outer_hits += 1
    if outer_hits > 0:
        adjacent_unbounded = 1

    return {
        "area": float(face.area),
        "perimeter": float(face.perimeter),
        "compactness": compactness(face.area, face.perimeter),
        "centroid_x": float(centroid.x),
        "centroid_y": float(centroid.y),
        "bbox_minx": float(minx),
        "bbox_miny": float(miny),
        "bbox_maxx": float(maxx),
        "bbox_maxy": float(maxy),
        "aspect_ratio": aspect_ratio(minx, miny, maxx, maxy),
        "number_of_doors": float(doors),
        "number_of_windows": float(windows),
        "wall_overlap": float(wall_overlap),
        "text_overlap": float(text_overlap),
        "adjacent_face_count": float(adjacent),
        "adjacent_unbounded": float(adjacent_unbounded),
        "boundary_complexity": float(len(list(geom.exterior.coords))),
        "schema_marker": 1.0,
        "feature_schema": 1.0,
    }


def feature_schema_version() -> str:
    return FACE_FEATURE_SCHEMA_VERSION
