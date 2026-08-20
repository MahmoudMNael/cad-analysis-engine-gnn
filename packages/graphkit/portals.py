from __future__ import annotations

from shapely.geometry import LineString, Point

from packages.common.configuration import Settings
from packages.common.geometry import canonicalize_unordered_pair
from packages.common.hashing import stable_hash
from packages.contracts.enums import EntityClass, PortalKind
from packages.graphkit.models import PlanGraph, Portal
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
    by_id = {p.id: p for p in primitives}
    for prim in primitives:
        score = entity_scores.get(prim.id)
        if score is None or score.predicted_class != EntityClass.OPENING.value:
            blob = " ".join(filter(None, [prim.layer, prim.block_name or ""])).lower()
            if "door" not in blob and "window" not in blob:
                continue
        geom = prim.geometry
        if geom.geom_type == "Polygon":
            minx, miny, maxx, maxy = geom.bounds
            line = LineString([(minx, (miny + maxy) / 2), (maxx, (miny + maxy) / 2)])
        elif geom.geom_type == "LineString":
            line = geom
        else:
            c = geom.centroid
            line = LineString([(c.x - 0.4, c.y), (c.x + 0.4, c.y)])
        kind = infer_portal_kind(prim, line.length, settings)
        nearest = _nearest_edge_ids(graph, line, settings)
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
        nearest = _nearest_edge_ids(graph, geom, settings)
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


def _nearest_edge_ids(graph: PlanGraph, geom: LineString, settings: Settings, limit: int = 2) -> list[str]:
    scored: list[tuple[float, str]] = []
    for edge in graph.edges:
        d = edge.geometry.distance(geom)
        if d <= max(settings.snap_tolerance_m * 4, 0.15):
            scored.append((d, edge.id))
    scored.sort()
    return [eid for _, eid in scored[:limit]]
