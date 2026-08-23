from __future__ import annotations

from collections.abc import Sequence

from shapely.ops import unary_union

from packages.common.errors import InvalidTargetError
from packages.contracts.dto.layout import RoomDto
from packages.views.models import MutableView, derived_room_id, polygon_from_room, room_from_polygon


def merge_rooms(view: MutableView, target_ids: Sequence[str]) -> RoomDto:
    members: list[RoomDto] = []
    seen: set[str] = set()
    for room_id in target_ids:
        room = view.rooms.get(room_id)
        if room is None:
            continue
        if room.id in seen:
            continue
        seen.add(room.id)
        members.append(room)
    if len(members) < 2:
        raise InvalidTargetError("merge_rooms requires at least two current derived rooms")
    geoms = [polygon_from_room(room) for room in members]
    merged = unary_union(geoms)
    if merged.geom_type == "MultiPolygon" and len(merged.geoms) == 1:
        merged = merged.geoms[0]
    if merged.geom_type != "Polygon":
        raise InvalidTargetError("rooms do not share a boundary that forms a single polygon")
    kinds = {room.kind for room in members}
    kind = "corridor" if "corridor" in kinds else next(iter(kinds))
    physical_ids = {room.physical_room_id for room in members if room.physical_room_id}
    physical_room_id = next(iter(physical_ids)) if len(physical_ids) == 1 else None
    names = [room.name for room in members if room.name]
    room_id = derived_room_id(merged, "|".join(sorted(seen)), "merge")
    room = room_from_polygon(
        merged,
        room_id=room_id,
        kind=kind,
        confidence=min(room.confidence for room in members),
        physical_room_id=physical_room_id,
        parent_room_id=None,
        name=names[0] if len(names) == 1 else None,
        door_ids=sorted({d for room in members for d in room.door_ids}),
        window_ids=sorted({w for room in members for w in room.window_ids}),
        object_ids=sorted({o for room in members for o in room.object_ids}),
    )
    for member in members:
        view.rooms.pop(member.id, None)
    view.rooms[room.id] = room
    return room
