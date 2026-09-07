"""Segment intersection detection and splitting (noding)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from packages.common.configuration import Settings
from packages.common.geometry import round_point
from packages.common.spatial_index import expand_bounds


@dataclass(frozen=True)
class RawSegment:
    primitive_id: str
    a: tuple[float, float]
    b: tuple[float, float]


def _point_on_segment(pt: tuple[float, float], a: tuple[float, float], b: tuple[float, float], tol: float) -> bool:
    line = LineString([a, b])
    d = line.distance(Point(pt))
    if d > tol:
        return False
    # exclude the exact endpoints (handled by snapping)
    if Point(pt).distance(Point(a)) <= tol or Point(pt).distance(Point(b)) <= tol:
        return False
    return True


def _split_segment(seg: RawSegment, points: list[tuple[float, float]], precision: int) -> list[RawSegment]:
    a, b = seg.a, seg.b
    unique = []
    seen = {round_point(a, precision), round_point(b, precision)}
    for pt in points:
        rpt = round_point(pt, precision)
        if rpt in seen:
            continue
        seen.add(rpt)
        unique.append(rpt)
    if not unique:
        if round_point(a, precision) == round_point(b, precision):
            return []
        return [RawSegment(seg.primitive_id, round_point(a, precision), round_point(b, precision))]
    dx, dy = b[0] - a[0], b[1] - a[1]
    denom = dx * dx + dy * dy or 1.0

    def key(pt: tuple[float, float]) -> float:
        return ((pt[0] - a[0]) * dx + (pt[1] - a[1]) * dy) / denom

    ordered = [round_point(a, precision)] + sorted(unique, key=key) + [round_point(b, precision)]
    out: list[RawSegment] = []
    for i in range(len(ordered) - 1):
        p, q = ordered[i], ordered[i + 1]
        if p == q:
            continue
        out.append(RawSegment(seg.primitive_id, p, q))
    return out


def node_segments(segments: list[RawSegment], settings: Settings) -> list[RawSegment]:
    if not segments:
        return []
    lines = [LineString([s.a, s.b]) for s in segments]
    tree = STRtree(lines)
    split_pts: list[list[tuple[float, float]]] = [[] for _ in segments]
    tol = settings.intersection_tolerance_m
    precision = settings.coord_precision

    for i, line in enumerate(lines):
        hits = tree.query(expand_bounds(line, tol), predicate="intersects")
        for j in hits:
            j = int(j)
            if j <= i:
                continue
            other = lines[j]
            inter = line.intersection(other)
            if inter.is_empty:
                continue
            pts: list[tuple[float, float]] = []
            if inter.geom_type == "Point":
                pts.append((float(inter.x), float(inter.y)))
            elif inter.geom_type == "MultiPoint":
                pts.extend((float(p.x), float(p.y)) for p in inter.geoms)
            elif inter.geom_type in {"LineString", "MultiLineString"}:
                pts.extend((float(x), float(y)) for x, y, *_ in inter.coords)
            for pt in pts:
                split_pts[i].append(pt)
                split_pts[j].append(pt)

        # T-junctions: endpoints of other segments landing on this interior
        for j in hits:
            j = int(j)
            if j == i:
                continue
            other = segments[j]
            for endpoint in (other.a, other.b):
                if _point_on_segment(endpoint, segments[i].a, segments[i].b, max(tol, settings.snap_tolerance_m)):
                    split_pts[i].append(endpoint)

    noded: list[RawSegment] = []
    for i, seg in enumerate(segments):
        noded.extend(_split_segment(seg, split_pts[i], precision))
    return noded
