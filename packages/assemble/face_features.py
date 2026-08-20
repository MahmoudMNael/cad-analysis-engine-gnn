from __future__ import annotations

from shapely.geometry import Point

from packages.assemble.polygonize import Face, FaceSet
from packages.common.geometry import aspect_ratio, compactness
from packages.graphkit.models import PlanGraph, Portal
from packages.ingest.models import Primitive
from packages.models_nn.features.face import FACE_FEATURE_SCHEMA_VERSION


def extract_face_features(
    face: Face,
    faces: FaceSet,
    graph: PlanGraph,
    portals: tuple[Portal, ...],
    texts: tuple[Primitive, ...],
    walls: tuple[Primitive, ...],
) -> dict[str, float]:
    geom = face.geometry
    minx, miny, maxx, maxy = geom.bounds
    centroid = geom.centroid
    doors = 0
    windows = 0
    for portal in portals:
        if portal.geometry.distance(geom) <= 1e-4 or geom.buffer(0.05).intersects(portal.geometry):
            if portal.kind.value == "door":
                doors += 1
            elif portal.kind.value == "window":
                windows += 1
    wall_overlap = 0.0
    for wall in walls:
        inter = wall.geometry.intersection(geom.boundary.buffer(0.05))
        wall_overlap += float(inter.length) if not inter.is_empty else 0.0
    text_overlap = 0.0
    for text in texts:
        if geom.intersects(text.geometry) or geom.contains(text.geometry.centroid):
            text_overlap += 1.0
    adjacent = 0
    adjacent_unbounded = 0
    for other in faces.faces:
        if other.id == face.id:
            continue
        if face.geometry.boundary.intersects(other.geometry.boundary.buffer(1e-4)):
            adjacent += 1
    # A face is adjacent to unbounded if some boundary edge is used once.
    boundary = geom.boundary
    shared = 0.0
    for edge in graph.edges:
        if boundary.distance(edge.geometry) <= 1e-3:
            shared += edge.geometry.length
    if shared < face.perimeter * 0.95:
        adjacent_unbounded = 1
    # Single enclosed rectangle: all edges are on the unbounded side → still indoor.
    # Use: if this face's envelope is not contained in union of others and has outer edges.
    outer_hits = 0
    for eid in faces.unbounded_edge_ids:
        edge = graph.edge_map().get(eid)
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
