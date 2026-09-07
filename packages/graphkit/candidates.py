"""Gap candidates from dangling/under-connected nodes only."""

from __future__ import annotations

import math

from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from packages.common.configuration import Settings
from packages.common.geometry import canonicalize_unordered_pair, orientation_sincos
from packages.common.hashing import stable_hash
from packages.common.spatial_index import SpatialIndex, expand_bounds
from packages.graphkit.models import GapCandidate, PlanGraph


def _pair_id(a: str, b: str, distance: float, precision: int) -> str:
    first, second = (a, b) if a < b else (b, a)
    return stable_hash(
        "gap_pair",
        [(0.0, round(distance, precision))],
        {"a": first, "b": second},
    )


def _collinear(graph: PlanGraph, a_id: str, b_id: str) -> bool:
    nodes = graph.node_map()
    edges = graph.edge_map()
    na, nb = nodes[a_id], nodes[b_id]
    dirs: list[tuple[float, float]] = []
    for nid in (a_id, b_id):
        node = nodes[nid]
        for eid in node.incident_edge_ids:
            edge = edges[eid]
            sin_a, cos_a = orientation_sincos(edge.geometry)
            dirs.append((cos_a, sin_a))
    if len(dirs) < 2:
        dx = nb.point[0] - na.point[0]
        dy = nb.point[1] - na.point[1]
        mag = math.hypot(dx, dy) or 1.0
        gap_dir = (dx / mag, dy / mag)
        if not dirs:
            return True
        return abs(dirs[0][0] * gap_dir[0] + dirs[0][1] * gap_dir[1]) > 0.85
    return abs(dirs[0][0] * dirs[1][0] + dirs[0][1] * dirs[1][1]) > 0.85


def generate_gap_candidates(graph: PlanGraph, settings: Settings) -> list[GapCandidate]:
    dangling = graph.dangling_node_ids()
    if len(dangling) < 2:
        return []
    nodes = graph.node_map()
    existing = {(e.node_a, e.node_b) for e in graph.edges} | {(e.node_b, e.node_a) for e in graph.edges}
    edge_lines = [e.geometry for e in graph.edges]
    tree = STRtree(edge_lines) if edge_lines else None
    out: list[GapCandidate] = []
    seen: set[tuple[str, str]] = set()
    d_max = settings.gap_max_distance_m
    node_index = SpatialIndex.build([(nid, Point(nodes[nid].point)) for nid in dangling])
    for i, a_id in enumerate(dangling):
        a = nodes[a_id]
        nearby = set(node_index.query(expand_bounds(Point(a.point), d_max)))
        for b_id in dangling[i + 1 :]:
            if b_id not in nearby:
                continue
            key = (a_id, b_id) if a_id < b_id else (b_id, a_id)
            if key in seen:
                continue
            seen.add(key)
            b = nodes[b_id]
            dist = math.hypot(a.point[0] - b.point[0], a.point[1] - b.point[1])
            if dist <= 0 or dist > d_max:
                continue
            already = key in existing or (key[1], key[0]) in existing
            if already:
                continue
            geom = LineString([a.point, b.point])
            crosses = False
            if tree is not None:
                hits = tree.query(geom, predicate="intersects")
                for h in hits:
                    other = edge_lines[int(h)]
                    inter = geom.intersection(other)
                    if inter.is_empty:
                        continue
                    if inter.geom_type == "Point":
                        pt = Point(inter.x, inter.y)
                        if pt.distance(Point(a.point)) > 1e-6 and pt.distance(Point(b.point)) > 1e-6:
                            crosses = True
                            break
                    else:
                        crosses = True
                        break
            collinear = _collinear(graph, a_id, b_id)
            opening_like = settings.door_width_min_m <= dist <= settings.door_width_max_m and collinear
            pair_id = _pair_id(a_id, b_id, dist, settings.coord_precision)
            out.append(
                GapCandidate(
                    pair_id=pair_id,
                    node_a=a_id,
                    node_b=b_id,
                    distance_m=dist,
                    geometry=geom,
                    collinear=collinear,
                    crosses=crosses,
                    already_connected=already,
                    opening_like=opening_like,
                )
            )
    return out
