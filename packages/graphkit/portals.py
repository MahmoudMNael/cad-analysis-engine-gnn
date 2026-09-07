from __future__ import annotations

import math

from shapely.geometry import LineString

from packages.common.configuration import Settings
from packages.common.geometry import canonicalize_unordered_pair
from packages.common.hashing import stable_hash
from packages.common.spatial_index import SpatialIndex, expand_bounds
from packages.contracts.enums import EntityClass, PortalKind
from packages.graphkit.models import GraphEdge, PlanGraph, Portal
from packages.ingest.models import Primitive
from packages.models_nn.common import EntityScore, GapScore


def _portal_id(kind: str, geom: LineString, precision: int) -> str:
    coords = canonicalize_unordered_pair(geom.coords[0], geom.coords[-1], precision)
    return stable_hash("portal", coords, {"kind": kind})


def infer_portal_kind(primitive: Primitive | None, distance_m: float, settings: Settings) -> PortalKind:
    blob = ""
    if primitive is not None:
        blob = " ".join(
            filter(None, [primitive.layer, primitive.block_name or "", primitive.source_entity_type])
        ).lower()
    if "window" in blob or "wnd" in blob or "glaz" in blob:
        return PortalKind.WINDOW
    if "door" in blob:
        return PortalKind.DOOR
    if settings.window_width_min_m <= distance_m <= settings.window_width_max_m and distance_m > settings.door_width_max_m:
        return PortalKind.WINDOW
    if settings.door_width_min_m <= distance_m <= settings.door_width_max_m:
        return PortalKind.DOOR
    return PortalKind.OPENING


def detect_portals(
    graph: PlanGraph,
    primitives: tuple[Primitive, ...],
    entity_scores: dict[str, EntityScore],
    gap_scores: dict[str, GapScore],
    gap_geoms: dict[str, LineString],
    settings: Settings,
) -> tuple[Portal, ...]:
    portals: list[Portal] = []
    edge_index = SpatialIndex.build([(e.id, e.geometry) for e in graph.edges])
    edge_by_id = graph.edge_map()
    for prim in primitives:
        score = entity_scores.get(prim.id)
        if score is None or score.predicted_class != EntityClass.OPENING.value:
            blob = " ".join(filter(None, [prim.layer, prim.block_name or ""])).lower()
            if "door" not in blob and "window" not in blob:
                continue
        line = project_opening_onto_walls(prim, edge_index, edge_by_id, settings)
        if line is None:
            continue
        kind = infer_portal_kind(prim, line.length, settings)
        nearest = _nearest_edge_ids(line, settings, edge_index, edge_by_id)
        portals.append(
            Portal(
                id=_portal_id(kind.value, line, settings.coord_precision),
                geometry=line,
                kind=kind,
                source_edge_ids=tuple(nearest),
                confidence=score.confidence if score else 0.6,
                source_primitive_ids=(prim.id,),
            )
        )
    for pair_id, score in gap_scores.items():
        if score.decision.value != "portal":
            continue
        geom = gap_geoms.get(pair_id)
        if geom is None:
            continue
        kind = infer_portal_kind(None, geom.length, settings)
        nearest = _nearest_edge_ids(geom, settings, edge_index, edge_by_id)
        portals.append(
            Portal(
                id=_portal_id(kind.value, geom, settings.coord_precision),
                geometry=geom,
                kind=kind,
                source_edge_ids=tuple(nearest),
                confidence=score.confidence,
            )
        )
    unique: dict[str, Portal] = {}
    for portal in portals:
        unique[portal.id] = portal
    return tuple(unique.values())


def project_opening_onto_walls(
    primitive: Primitive,
    edge_index: SpatialIndex,
    edge_by_id: dict[str, GraphEdge],
    settings: Settings,
) -> LineString | None:
    """Map door/window artwork onto the supporting wall. Never keep interior swing chords."""
    if not edge_by_id:
        return None
    geom = primitive.geometry
    search_r = max(settings.door_width_max_m, 1.25)
    nearest_id = None
    nearest_d = search_r + 1.0
    for eid in edge_index.query(expand_bounds(geom, search_r)):
        edge = edge_by_id.get(eid)
        if edge is None or edge.geometry.is_empty:
            continue
        d = edge.geometry.distance(geom)
        if d < nearest_d:
            nearest_d = d
            nearest_id = eid
    if nearest_id is None or nearest_d > search_r:
        return None
    edge = edge_by_id[nearest_id]
    frame = _line_frame(edge.geometry)
    if frame is None:
        return None
    origin, direction, _length = frame
    on_line_tol = max(settings.snap_tolerance_m * 8.0, 0.30)
    params: list[float] = []
    for x, y in _opening_samples(primitive):
        t, dist = _project_to_line(x, y, origin, direction)
        if dist <= on_line_tol:
            params.append(t)
    if len(params) < 2:
        return None
    t0, t1 = min(params), max(params)
    span = t1 - t0
    if span < settings.window_width_min_m:
        return None
    if span > settings.window_width_max_m:
        return None
    a = (origin[0] + direction[0] * t0, origin[1] + direction[1] * t0)
    b = (origin[0] + direction[0] * t1, origin[1] + direction[1] * t1)
    line = LineString(canonicalize_unordered_pair(a, b, settings.coord_precision))
    if line.length < settings.window_width_min_m:
        return None
    return line


def _opening_samples(primitive: Primitive) -> list[tuple[float, float]]:
    geom = primitive.geometry
    samples: list[tuple[float, float]] = []
    center = primitive.attributes.get("center")
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        samples.append((float(center[0]), float(center[1])))
    if geom.geom_type == "LineString" and len(geom.coords) >= 2:
        coords = list(geom.coords)
        samples.append((float(coords[0][0]), float(coords[0][1])))
        samples.append((float(coords[-1][0]), float(coords[-1][1])))
        mid = coords[len(coords) // 2]
        samples.append((float(mid[0]), float(mid[1])))
    elif geom.geom_type == "Polygon":
        minx, miny, maxx, maxy = geom.bounds
        samples.extend(
            [
                (minx, miny),
                (maxx, miny),
                (maxx, maxy),
                (minx, maxy),
                ((minx + maxx) / 2.0, (miny + maxy) / 2.0),
            ]
        )
    else:
        c = geom.centroid
        samples.append((float(c.x), float(c.y)))
    return samples


def _line_frame(geom: LineString) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    coords = list(geom.coords)
    if len(coords) < 2:
        return None
    x0, y0 = float(coords[0][0]), float(coords[0][1])
    x1, y1 = float(coords[-1][0]), float(coords[-1][1])
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return None
    return (x0, y0), (dx / length, dy / length), length


def _project_to_line(
    x: float,
    y: float,
    origin: tuple[float, float],
    direction: tuple[float, float],
) -> tuple[float, float]:
    t = (x - origin[0]) * direction[0] + (y - origin[1]) * direction[1]
    px = origin[0] + direction[0] * t
    py = origin[1] + direction[1] * t
    dist = math.hypot(x - px, y - py)
    return t, dist


def _nearest_edge_ids(
    geom: LineString,
    settings: Settings,
    edge_index: SpatialIndex,
    edge_by_id: dict[str, GraphEdge],
    limit: int = 2,
) -> list[str]:
    radius = max(settings.snap_tolerance_m * 4, 0.15)
    scored: list[tuple[float, str]] = []
    for eid in edge_index.query(expand_bounds(geom, radius)):
        edge = edge_by_id.get(eid)
        if edge is None:
            continue
        d = edge.geometry.distance(geom)
        if d <= radius:
            scored.append((d, edge.id))
    scored.sort()
    return [eid for _, eid in scored[:limit]]
