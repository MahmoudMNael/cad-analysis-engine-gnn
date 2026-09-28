from __future__ import annotations

import math
from collections.abc import Sequence

from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import split as shapely_split

from packages.contracts.dto.layout import RoomDto, VirtualDividerDto
from packages.views.models import (
    MutableView,
    derived_room_id,
    make_divider,
    polygon_from_room,
    room_from_polygon,
)

_INTERIOR_EPS = 1e-6
_BLADE_PAD_M = 1e-3


def _extend_line(line: LineString, polygon: Polygon) -> LineString:
    coords = list(line.coords)
    x0, y0 = float(coords[0][0]), float(coords[0][1])
    x1, y1 = float(coords[-1][0]), float(coords[-1][1])
    dx, dy = x1 - x0, y1 - y0
    mag = math.hypot(dx, dy)
    if mag <= 0:
        return line
    minx, miny, maxx, maxy = polygon.bounds
    pad = max(maxx - minx, maxy - miny, mag, 1.0) * 4.0
    ux, uy = dx / mag, dy / mag
    return LineString([(x0 - ux * pad, y0 - uy * pad), (x1 + ux * pad, y1 + uy * pad)])


def _nudge_line(line: LineString, pad: float = _BLADE_PAD_M) -> LineString:
    coords = list(line.coords)
    x0, y0 = float(coords[0][0]), float(coords[0][1])
    x1, y1 = float(coords[-1][0]), float(coords[-1][1])
    dx, dy = x1 - x0, y1 - y0
    mag = math.hypot(dx, dy)
    if mag <= 0:
        return line
    ux, uy = dx / mag, dy / mag
    return LineString([(x0 - ux * pad, y0 - uy * pad), (x1 + ux * pad, y1 + uy * pad)])


def _line_pieces(geom: BaseGeometry) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString" and len(geom.coords) >= 2:
        return [geom]
    if geom.geom_type == "LinearRing":
        return [LineString(geom.coords)]
    if geom.geom_type in {"MultiLineString", "GeometryCollection", "MultiPolygon"}:
        out: list[LineString] = []
        for part in geom.geoms:
            out.extend(_line_pieces(part))
        return out
    return []


def _pick_stroke_piece(pieces: Sequence[LineString], stroke: LineString) -> LineString | None:
    if not pieces:
        return None
    scored: list[tuple[float, float, float, LineString]] = []
    for piece in pieces:
        if piece.is_empty or piece.length <= 0:
            continue
        overlap = piece.intersection(stroke)
        overlap_len = float(overlap.length) if not overlap.is_empty else 0.0
        dist = float(piece.distance(stroke))
        scored.append((overlap_len, -dist, piece.length, piece))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][3]


def _subtract_islands(target: Polygon, others: Sequence[Polygon], stroke: LineString) -> Polygon:
    result: BaseGeometry = target
    for other in others:
        if other.is_empty or other.area <= 0:
            continue
        interior = other.buffer(-_INTERIOR_EPS)
        if interior.is_empty:
            continue
        if not result.contains(interior):
            continue
        result = result.difference(other)
    if result.is_empty:
        return target
    if result.geom_type == "Polygon":
        return result
    if result.geom_type == "MultiPolygon":
        parts = [part for part in result.geoms if part.geom_type == "Polygon" and part.area > 1e-9]
        if not parts:
            return target
        hitting = [part for part in parts if part.intersects(stroke) or part.distance(stroke) <= _INTERIOR_EPS]
        return max(hitting or parts, key=lambda part: part.area)
    return target


def _local_chord(
    polygon: Polygon,
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    stroke = LineString([start, end])
    if polygon.is_empty or stroke.is_empty:
        return None
    extended = _extend_line(stroke, polygon)
    pieces = _line_pieces(extended.intersection(polygon))
    piece = _pick_stroke_piece(pieces, stroke)
    if piece is None:
        return None
    coords = list(piece.coords)
    return (float(coords[0][0]), float(coords[0][1])), (float(coords[-1][0]), float(coords[-1][1]))


def split_polygon(polygon: Polygon, start: tuple[float, float], end: tuple[float, float]) -> list[Polygon]:
    chord = _local_chord(polygon, start, end)
    if chord is None:
        return []
    blade = _nudge_line(LineString([chord[0], chord[1]]))
    result = shapely_split(polygon, blade)
    pieces: list[Polygon] = []
    geoms = list(result.geoms) if hasattr(result, "geoms") else [result]
    for geom in geoms:
        if geom.geom_type == "Polygon" and geom.area > 1e-9:
            pieces.append(geom)
        elif geom.geom_type == "MultiPolygon":
            pieces.extend(part for part in geom.geoms if part.area > 1e-9)
    if len(pieces) < 2:
        return []
    pieces.sort(key=lambda g: (round(g.centroid.x, 6), round(g.centroid.y, 6), round(g.area, 6)))
    return pieces


def _stroke_enters_interior(polygon: Polygon, start: tuple[float, float], end: tuple[float, float]) -> bool:
    stroke = LineString([start, end])
    if polygon.is_empty or stroke.is_empty:
        return False
    interior = polygon.buffer(-_INTERIOR_EPS)
    if interior.is_empty:
        inter = polygon.intersection(stroke)
        return (not inter.is_empty) and float(getattr(inter, "length", 0.0) or 0.0) > 0.0
    return bool(interior.intersects(stroke))


def _auto_target_ids(
    rooms: Sequence[RoomDto],
    start: tuple[float, float],
    end: tuple[float, float],
) -> list[str]:
    """Choose the single room the drawn segment actually goes through."""
    stroke = LineString([start, end])
    mid = Point((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    ranked: list[tuple[bool, float, str]] = []
    for room in rooms:
        polygon = polygon_from_room(room)
        if not _stroke_enters_interior(polygon, start, end):
            continue
        overlap = polygon.intersection(stroke)
        overlap_len = float(overlap.length) if not overlap.is_empty else 0.0
        covers_mid = bool(polygon.buffer(_INTERIOR_EPS).covers(mid))
        ranked.append((covers_mid, overlap_len, room.id))
    if not ranked:
        return []
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [ranked[0][2]]


def _prepared_polygon(
    room: RoomDto,
    rooms: Sequence[RoomDto],
    start: tuple[float, float],
    end: tuple[float, float],
) -> Polygon:
    target = polygon_from_room(room)
    others = [polygon_from_room(other) for other in rooms if other.id != room.id]
    return _subtract_islands(target, others, LineString([start, end]))


def rooms_split_by_line(
    rooms: Sequence[RoomDto],
    start: tuple[float, float],
    end: tuple[float, float],
    target_ids: Sequence[str] | None = None,
) -> dict[str, list[Polygon]]:
    if target_ids:
        wanted = set(target_ids)
    else:
        wanted = set(_auto_target_ids(rooms, start, end))
        if not wanted:
            return {}
    out: dict[str, list[Polygon]] = {}
    for room in rooms:
        if room.id not in wanted and room.physical_room_id not in wanted:
            continue
        pieces = split_polygon(_prepared_polygon(room, rooms, start, end), start, end)
        if len(pieces) >= 2:
            out[room.id] = pieces
    return out


def split_room(
    view: MutableView,
    start: tuple[float, float],
    end: tuple[float, float],
    target_ids: Sequence[str] | None = None,
) -> tuple[list[str], VirtualDividerDto]:
    rooms = list(view.rooms.values())
    split_map = rooms_split_by_line(rooms, start, end, target_ids)
    clip_ids = list(split_map.keys())
    if not clip_ids:
        if target_ids:
            clip_ids = [tid for tid in target_ids if tid in view.rooms]
        else:
            clip_ids = _auto_target_ids(rooms, start, end)
    if len(clip_ids) == 1 and clip_ids[0] in view.rooms:
        parent = view.rooms[clip_ids[0]]
        chord = _local_chord(_prepared_polygon(parent, rooms, start, end), start, end)
        if chord is not None:
            start, end = chord
    divider = make_divider(start, end)
    view.dividers[divider.id] = divider
    created: list[str] = []
    for room_id, pieces in split_map.items():
        parent = view.rooms.pop(room_id)
        for piece in pieces:
            child_id = derived_room_id(piece, parent.id, "split")
            child = room_from_polygon(
                piece,
                room_id=child_id,
                kind=parent.kind,
                confidence=parent.confidence,
                physical_room_id=parent.physical_room_id or parent.id,
                parent_room_id=parent.id,
                name=parent.name,
                door_ids=list(parent.door_ids),
                window_ids=list(parent.window_ids),
                object_ids=[],
            )
            view.rooms[child.id] = child
            created.append(child.id)
    return created, divider
