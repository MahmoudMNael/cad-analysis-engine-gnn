from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from shapely.geometry import Polygon

from packages.common.geometry import bbox_tuple, canonicalize_polygon_coords, canonicalize_unordered_pair
from packages.common.hashing import stable_hash
from packages.contracts.dto.common import BBoxDto, PointDto
from packages.contracts.dto.layout import DimensionsDto, FaceDto, RoomDto, VirtualDividerDto
from packages.contracts.enums import FeedbackAction

COORD_PRECISION = 6
OBJECT_TOLERANCE_M = 0.05


@dataclass(frozen=True)
class ViewEvent:
    event_id: str
    sequence: int
    action: FeedbackAction | str
    target_ids: tuple[str, ...]
    payload: Mapping[str, Any]
    layout_rev: int
    retracted_by_event_id: str | None = None
    actor: str = "user"

    @property
    def action_value(self) -> str:
        if isinstance(self.action, FeedbackAction):
            return self.action.value
        return str(self.action)


def coerce_event(item: Any) -> ViewEvent:
    if isinstance(item, ViewEvent):
        return item
    if hasattr(item, "model_dump"):
        data = item.model_dump()
        event_id = str(data.get("event_id") or data.get("id") or "")
        return ViewEvent(
            event_id=event_id,
            sequence=int(data.get("sequence") or 0),
            action=data.get("action") or "",
            target_ids=tuple(str(x) for x in (data.get("target_ids") or ())),
            payload=dict(data.get("payload") or data.get("payload_json") or {}),
            layout_rev=int(data.get("layout_rev") or 0),
            retracted_by_event_id=data.get("retracted_by_event_id"),
            actor=str(data.get("actor") or "user"),
        )
    data = dict(item)
    return ViewEvent(
        event_id=str(data.get("event_id") or data.get("id") or ""),
        sequence=int(data.get("sequence") or 0),
        action=data.get("action") or "",
        target_ids=tuple(str(x) for x in (data.get("target_ids") or ())),
        payload=dict(data.get("payload") or data.get("payload_json") or {}),
        layout_rev=int(data.get("layout_rev") or 0),
        retracted_by_event_id=data.get("retracted_by_event_id"),
        actor=str(data.get("actor") or "user"),
    )


def polygon_from_room(room: RoomDto) -> Polygon:
    coords = [(p.x, p.y) for p in room.boundary]
    if coords[0] != coords[-1]:
        coords = [*coords, coords[0]]
    return Polygon(coords)


def polygon_from_face(face: FaceDto) -> Polygon:
    coords = [(p.x, p.y) for p in face.boundary]
    if coords[0] != coords[-1]:
        coords = [*coords, coords[0]]
    return Polygon(coords)


def ring_points(polygon: Polygon) -> list[PointDto]:
    coords = list(polygon.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [PointDto(x=float(x), y=float(y)) for x, y, *_ in coords]


def room_from_polygon(
    polygon: Polygon,
    *,
    room_id: str,
    kind: str,
    confidence: float,
    physical_room_id: str | None,
    parent_room_id: str | None,
    name: str | None = None,
    selectable: bool = True,
    neighbor_ids: Sequence[str] = (),
    door_ids: Sequence[str] = (),
    window_ids: Sequence[str] = (),
    object_ids: Sequence[str] = (),
) -> RoomDto:
    minx, miny, maxx, maxy = bbox_tuple(polygon)
    centroid = polygon.centroid
    return RoomDto(
        id=room_id,
        boundary=ring_points(polygon),
        area_m2=float(polygon.area),
        area_net_m2=float(polygon.area),
        area_gross_m2=float(polygon.area),
        center=PointDto(x=float(centroid.x), y=float(centroid.y)),
        perimeter_m=float(polygon.length),
        bbox=BBoxDto(min_x=minx, min_y=miny, max_x=maxx, max_y=maxy),
        confidence=confidence,
        name=name,
        kind=kind,
        selectable=selectable,
        neighbor_ids=list(neighbor_ids),
        door_ids=list(door_ids),
        window_ids=list(window_ids),
        object_ids=list(object_ids),
        physical_room_id=physical_room_id,
        parent_room_id=parent_room_id,
        dimensions=DimensionsDto(
            width_m=max(maxx - minx, 0.0),
            length_m=max(maxy - miny, 0.0),
        ),
    )


def derived_room_id(polygon: Polygon, parent_id: str, kind: str) -> str:
    coords = canonicalize_polygon_coords(list(polygon.exterior.coords), COORD_PRECISION)
    return stable_hash("derived_room", coords, {"parent": parent_id, "kind": kind})


def divider_id(start: tuple[float, float], end: tuple[float, float]) -> str:
    pair = canonicalize_unordered_pair(start, end, COORD_PRECISION)
    return stable_hash("virtual_divider", pair, {})


def make_divider(start: tuple[float, float], end: tuple[float, float]) -> VirtualDividerDto:
    return VirtualDividerDto(
        id=divider_id(start, end),
        start=PointDto(x=float(start[0]), y=float(start[1])),
        end=PointDto(x=float(end[0]), y=float(end[1])),
    )


def payload_point_pair(payload: Mapping[str, Any]) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if "start_x" in payload and "end_x" in payload:
        return (
            (float(payload["start_x"]), float(payload["start_y"])),
            (float(payload["end_x"]), float(payload["end_y"])),
        )
    start = payload.get("start")
    end = payload.get("end")
    if isinstance(start, Mapping) and isinstance(end, Mapping):
        return (
            (float(start["x"]), float(start["y"])),
            (float(end["x"]), float(end["y"])),
        )
    node_a = payload.get("node_a_point")
    node_b = payload.get("node_b_point")
    if isinstance(node_a, Sequence) and isinstance(node_b, Sequence) and len(node_a) >= 2 and len(node_b) >= 2:
        return ((float(node_a[0]), float(node_a[1])), (float(node_b[0]), float(node_b[1])))
    return None


def recompute_neighbors(rooms: Sequence[RoomDto]) -> list[RoomDto]:
    polys = [polygon_from_room(room) for room in rooms]
    neighbor_map: dict[str, list[str]] = {room.id: [] for room in rooms}
    for i, a in enumerate(rooms):
        for j in range(i + 1, len(rooms)):
            b = rooms[j]
            if polys[i].boundary.intersects(polys[j].boundary.buffer(1e-4)):
                neighbor_map[a.id].append(b.id)
                neighbor_map[b.id].append(a.id)
    updated: list[RoomDto] = []
    for room in rooms:
        dumped = room.model_dump()
        dumped["neighbor_ids"] = sorted(neighbor_map[room.id])
        updated.append(RoomDto.model_validate(dumped))
    return updated


@dataclass
class MutableView:
    rooms: dict[str, RoomDto] = field(default_factory=dict)
    dividers: dict[str, VirtualDividerDto] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
