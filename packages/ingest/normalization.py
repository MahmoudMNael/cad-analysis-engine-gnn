"""Normalize extracted geometry after conversion to metres."""

from __future__ import annotations

from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid

from packages.common.geometry import (
    canonicalize_linestring_coords,
    canonicalize_polygon_coords,
    finite_geometry,
    round_point,
)


def normalize_geometry(geom: BaseGeometry, precision: int) -> BaseGeometry | None:
    if geom.geom_type == "LineString" and geom.length == 0:
        return None
    if geom.is_empty or not finite_geometry(geom):
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
        if geom.is_empty:
            return None
    if geom.geom_type == "Point":
        x, y = round_point((geom.x, geom.y), precision)
        return Point(x, y)
    if geom.geom_type == "LineString":
        pts = canonicalize_linestring_coords(list(geom.coords), precision)
        if len(pts) < 2:
            return None
        line = LineString(pts)
        return None if line.length == 0 else line
    if geom.geom_type == "Polygon":
        ext = canonicalize_polygon_coords(list(geom.exterior.coords), precision)
        if len(ext) < 3:
            return None
        holes = [
            canonicalize_polygon_coords(list(ring.coords), precision)
            for ring in geom.interiors
        ]
        poly = Polygon(ext, [h for h in holes if len(h) >= 3])
        if poly.is_empty or poly.area == 0:
            return None
        return poly if poly.is_valid else make_valid(poly)
    if geom.geom_type == "MultiLineString":
        parts = [normalize_geometry(p, precision) for p in geom.geoms]
        lines = [p for p in parts if p is not None and p.geom_type == "LineString"]
        if not lines:
            return None
        if len(lines) == 1:
            return lines[0]
        from shapely.geometry import MultiLineString

        return MultiLineString(lines)
    return geom
