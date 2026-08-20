"""Shared deterministic geometry helpers. Coordinates are metres."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid


def is_finite_number(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def round_coord(value: float, precision: int) -> float:
    return round(float(value), precision)


def round_point(point: Sequence[float], precision: int) -> tuple[float, float]:
    return (round_coord(point[0], precision), round_coord(point[1], precision))


def coords_2d(geom: BaseGeometry) -> list[tuple[float, float]]:
    if geom.is_empty:
        return []
    if geom.geom_type == "Point":
        return [(float(geom.x), float(geom.y))]
    if geom.geom_type in {"LineString", "LinearRing"}:
        return [(float(x), float(y)) for x, y, *_ in geom.coords]
    if geom.geom_type == "Polygon":
        return [(float(x), float(y)) for x, y, *_ in geom.exterior.coords]
    if hasattr(geom, "geoms"):
        out: list[tuple[float, float]] = []
        for part in geom.geoms:
            out.extend(coords_2d(part))
        return out
    return []


def drop_consecutive_duplicates(
    points: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    if not points:
        return []
    out = [points[0]]
    for pt in points[1:]:
        if pt != out[-1]:
            out.append(pt)
    return out


def canonicalize_linestring_coords(
    points: Sequence[Sequence[float]],
    precision: int,
) -> list[tuple[float, float]]:
    pts = drop_consecutive_duplicates([round_point(p, precision) for p in points])
    if len(pts) < 2:
        return pts
    if pts[0] > pts[-1]:
        pts = list(reversed(pts))
    return pts


def _ring_area(points: Sequence[tuple[float, float]]) -> float:
    area = 0.0
    n = len(points)
    if n < 3:
        return 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def canonicalize_polygon_coords(
    points: Sequence[Sequence[float]],
    precision: int,
) -> list[tuple[float, float]]:
    pts = [round_point(p, precision) for p in points]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    pts = drop_consecutive_duplicates(pts)
    if len(pts) < 3:
        return pts
    if _ring_area(pts) < 0:
        pts = list(reversed(pts))
    min_i = min(range(len(pts)), key=lambda i: (pts[i][0], pts[i][1]))
    pts = pts[min_i:] + pts[:min_i]
    return pts


def canonicalize_unordered_pair(
    a: Sequence[float],
    b: Sequence[float],
    precision: int,
) -> list[tuple[float, float]]:
    pa = round_point(a, precision)
    pb = round_point(b, precision)
    return [pa, pb] if pa <= pb else [pb, pa]


def geometry_hash_payload(kind: str, geom: BaseGeometry, precision: int) -> list[tuple[float, float]]:
    if geom.is_empty:
        return []
    if geom.geom_type == "Point":
        return [round_point((geom.x, geom.y), precision)]
    if geom.geom_type in {"LineString", "LinearRing"}:
        return canonicalize_linestring_coords(list(geom.coords), precision)
    if geom.geom_type == "Polygon":
        return canonicalize_polygon_coords(list(geom.exterior.coords), precision)
    pts = coords_2d(geom)
    return canonicalize_linestring_coords(pts, precision) if pts else []


def bbox_tuple(geom: BaseGeometry) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = geom.bounds
    return (float(minx), float(miny), float(maxx), float(maxy))


def length_of(geom: BaseGeometry) -> float:
    if geom.is_empty:
        return 0.0
    if geom.geom_type in {"Polygon", "MultiPolygon"}:
        return float(geom.length)
    return float(geom.length)


def finite_geometry(geom: BaseGeometry) -> bool:
    if geom.is_empty:
        return False
    for x, y in coords_2d(geom):
        if not is_finite_number(x) or not is_finite_number(y):
            return False
    return True


def orientation_sincos(geom: BaseGeometry) -> tuple[float, float]:
    if geom.geom_type in {"LineString", "LinearRing"} and len(geom.coords) >= 2:
        x0, y0 = geom.coords[0][0], geom.coords[0][1]
        x1, y1 = geom.coords[-1][0], geom.coords[-1][1]
        dx, dy = x1 - x0, y1 - y0
        mag = math.hypot(dx, dy)
        if mag <= 0:
            return 0.0, 1.0
        return dy / mag, dx / mag
    return 0.0, 1.0


def compactness(area: float, perimeter: float) -> float:
    if perimeter <= 0 or area <= 0:
        return 0.0
    return float(4.0 * math.pi * area / (perimeter * perimeter))


def aspect_ratio(minx: float, miny: float, maxx: float, maxy: float) -> float:
    w = max(maxx - minx, 1e-9)
    h = max(maxy - miny, 1e-9)
    return float(max(w, h) / min(w, h))


def repair_polygon(geom: BaseGeometry) -> tuple[BaseGeometry, bool]:
    if geom.is_valid:
        return geom, False
    repaired = make_valid(geom)
    return repaired, True


def as_linestring(a: Sequence[float], b: Sequence[float]) -> LineString:
    return LineString([(float(a[0]), float(a[1])), (float(b[0]), float(b[1]))])


def centroid_point(geom: BaseGeometry) -> Point:
    c = geom.centroid
    return Point(float(c.x), float(c.y))


def iter_line_segments(geom: BaseGeometry) -> Iterable[tuple[tuple[float, float], tuple[float, float]]]:
    if geom.is_empty:
        return
    if geom.geom_type == "LineString":
        coords = [(float(x), float(y)) for x, y, *_ in geom.coords]
        for i in range(len(coords) - 1):
            if coords[i] != coords[i + 1]:
                yield coords[i], coords[i + 1]
        return
    if geom.geom_type == "MultiLineString":
        for part in geom.geoms:
            yield from iter_line_segments(part)
        return
    if geom.geom_type == "Polygon":
        yield from iter_line_segments(LineString(geom.exterior.coords))
        return
    if geom.geom_type == "LinearRing":
        yield from iter_line_segments(LineString(geom.coords))


def scale_geometry(geom: BaseGeometry, factor: float) -> BaseGeometry:
    if factor == 1.0:
        return geom
    from shapely.affinity import scale

    return scale(geom, xfact=factor, yfact=factor, zfact=1.0, origin=(0.0, 0.0, 0.0))
