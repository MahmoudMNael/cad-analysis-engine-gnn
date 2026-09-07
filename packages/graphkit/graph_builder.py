from __future__ import annotations

from collections import defaultdict

from shapely.geometry import LineString

from packages.common.configuration import Settings
from packages.common.geometry import canonicalize_unordered_pair, chord_curvature, iter_line_segments
from packages.common.hashing import stable_hash
from packages.contracts.enums import EntityClass
from packages.graphkit.models import GraphEdge, GraphNode, PlanGraph, Portal
from packages.graphkit.noding import RawSegment, node_segments
from packages.graphkit.snapping import snap_endpoints
from packages.graphkit.validation import validate_plan_graph
from packages.ingest.models import Primitive
from packages.models_nn.common import EntityScore, GapScore

NON_WALL_CLASSES = {
    EntityClass.FURNITURE.value,
    EntityClass.TEXT.value,
    EntityClass.DIM.value,
    EntityClass.HATCH.value,
    EntityClass.OPENING.value,
}
ARC_KINDS = {"arc", "circle"}
ARC_SOURCE_TYPES = {"ARC", "CIRCLE", "ELLIPSE"}


def _node_id(point: tuple[float, float], precision: int) -> str:
    return stable_hash("node", [point], {"precision": precision})


def _edge_id(a: tuple[float, float], b: tuple[float, float], precision: int) -> str:
    pair = canonicalize_unordered_pair(a, b, precision)
    return stable_hash("edge", pair, {})


def _named_wall(primitive: Primitive) -> bool:
    blob = f"{primitive.layer} {primitive.block_name or ''}".lower()
    return any(word in blob for word in ("wall", "a-wall", "mur", "partition"))


def _opening_artwork(primitive: Primitive) -> bool:
    if primitive.kind.value in ARC_KINDS or primitive.source_entity_type.upper() in ARC_SOURCE_TYPES:
        return True
    geom = primitive.geometry
    if geom.geom_type != "LineString" or len(geom.coords) < 3:
        return False
    start, end = geom.coords[0], geom.coords[-1]
    closed = bool(primitive.attributes.get("closed")) or (start[0] == end[0] and start[1] == end[1])
    if closed:
        return False
    return chord_curvature(geom) >= 0.08 and 0.3 <= primitive.length <= 3.0


def wall_candidate_primitives(
    primitives: tuple[Primitive, ...],
    scores: dict[str, EntityScore],
    settings: Settings,
) -> list[Primitive]:
    out = []
    for prim in primitives:
        if prim.attributes.get("measurement"):
            continue
        if prim.kind.value in {"text", "hatch", "circle"}:
            continue
        score = scores.get(prim.id)
        if score is None:
            continue
        if score.predicted_class in NON_WALL_CLASSES:
            continue
        if _opening_artwork(prim) and not _named_wall(prim):
            continue
        wall_p = score.probabilities.get(EntityClass.WALL.value, 0.0)
        if score.predicted_class == EntityClass.WALL.value or wall_p >= settings.entity_threshold:
            out.append(prim)
    return out


def build_plan_graph(
    primitives: tuple[Primitive, ...],
    scores: dict[str, EntityScore],
    settings: Settings,
) -> PlanGraph:
    walls = wall_candidate_primitives(primitives, scores, settings)
    raw: list[RawSegment] = []
    points: list[tuple[float, float]] = []
    for prim in walls:
        for a, b in iter_line_segments(prim.geometry):
            if LineString([a, b]).length < settings.min_segment_length_m:
                continue
            raw.append(RawSegment(prim.id, a, b))
            points.extend([a, b])
    snapped = snap_endpoints(points, settings)
    snapped_segs: list[RawSegment] = []
    idx = 0
    for seg in raw:
        a = snapped[idx]
        b = snapped[idx + 1]
        idx += 2
        if a == b:
            continue
        if LineString([a, b]).length < settings.min_segment_length_m:
            continue
        snapped_segs.append(RawSegment(seg.primitive_id, a, b))
    noded = node_segments(snapped_segs, settings)

    incident_prims: dict[tuple[float, float], set[str]] = defaultdict(set)
    grouped: dict[tuple[tuple[float, float], tuple[float, float]], list[str]] = defaultdict(list)
    for seg in noded:
        a, b = (seg.a, seg.b) if seg.a <= seg.b else (seg.b, seg.a)
        if a == b:
            continue
        grouped[(a, b)].append(seg.primitive_id)
        incident_prims[a].add(seg.primitive_id)
        incident_prims[b].add(seg.primitive_id)

    nodes_tmp: dict[tuple[float, float], str] = {}
    for pt in incident_prims:
        nodes_tmp[pt] = _node_id(pt, settings.coord_precision)

    edges: list[GraphEdge] = []
    incident_edges: dict[str, list[str]] = defaultdict(list)
    for (a, b), prim_ids in grouped.items():
        eid = _edge_id(a, b, settings.coord_precision)
        na, nb = nodes_tmp[a], nodes_tmp[b]
        conf = 1.0
        if prim_ids:
            conf = max(scores[p].confidence for p in prim_ids if p in scores) if any(p in scores for p in prim_ids) else 0.8
        edges.append(
            GraphEdge(
                id=eid,
                node_a=na,
                node_b=nb,
                geometry=LineString([a, b]),
                primitive_ids=tuple(sorted(set(prim_ids))),
                class_prior=EntityClass.WALL.value,
                predicted_class=EntityClass.WALL.value,
                confidence=conf,
            )
        )
        incident_edges[na].append(eid)
        incident_edges[nb].append(eid)

    nodes = [
        GraphNode(
            id=nid,
            point=pt,
            incident_edge_ids=tuple(sorted(incident_edges.get(nid, []))),
            source_primitive_ids=tuple(sorted(incident_prims[pt])),
        )
        for pt, nid in nodes_tmp.items()
    ]
    graph = PlanGraph(nodes=tuple(nodes), edges=tuple(edges), portals=(), metadata={"wall_primitives": len(walls)})
    errors = validate_plan_graph(graph)
    if errors:
        graph = PlanGraph(
            nodes=graph.nodes,
            edges=graph.edges,
            portals=graph.portals,
            metadata={**dict(graph.metadata), "validation_errors": errors},
        )
    return graph


def close_graph(
    graph: PlanGraph,
    gap_scores: dict[str, GapScore],
    candidates_by_id: dict[str, object],
    settings: Settings,
    extra_portals: tuple[Portal, ...] = (),
) -> PlanGraph:
    from packages.graphkit.models import GapCandidate

    nodes = {n.id: n for n in graph.nodes}
    edges = list(graph.edges)
    incident: dict[str, list[str]] = {n.id: list(n.incident_edge_ids) for n in graph.nodes}
    existing = {(e.node_a, e.node_b) for e in edges} | {(e.node_b, e.node_a) for e in edges}

    for pair_id, score in gap_scores.items():
        cand = candidates_by_id.get(pair_id)
        if not isinstance(cand, GapCandidate):
            continue
        if score.decision.value not in {"close", "portal"}:
            continue
        key = (cand.node_a, cand.node_b)
        if key in existing or (key[1], key[0]) in existing:
            continue
        if cand.crosses or cand.distance_m > settings.gap_max_distance_m:
            continue
        a = nodes[cand.node_a]
        b = nodes[cand.node_b]
        eid = _edge_id(a.point, b.point, settings.coord_precision)
        predicted = "portal" if score.decision.value == "portal" else EntityClass.WALL.value
        edge = GraphEdge(
            id=eid,
            node_a=a.id,
            node_b=b.id,
            geometry=LineString([a.point, b.point]),
            primitive_ids=(),
            class_prior=predicted,
            predicted_class=predicted,
            confidence=score.confidence,
        )
        edges.append(edge)
        incident[a.id].append(eid)
        incident[b.id].append(eid)
        existing.add(key)

    new_nodes = tuple(
        GraphNode(
            id=n.id,
            point=n.point,
            incident_edge_ids=tuple(sorted(set(incident.get(n.id, [])))),
            source_primitive_ids=n.source_primitive_ids,
            attributes=n.attributes,
        )
        for n in graph.nodes
    )
    return PlanGraph(
        nodes=new_nodes,
        edges=tuple(edges),
        portals=extra_portals,
        metadata=dict(graph.metadata),
    )
