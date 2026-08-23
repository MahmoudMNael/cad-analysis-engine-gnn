"""Pure derived-view reducer. Never mutates detector snapshot objects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from shapely.geometry import Point

from packages.contracts.dto.layout import LayoutDto, ObjectDto, RoomDto, TextDto
from packages.contracts.enums import FeedbackAction
from packages.views.delete import delete_room
from packages.views.merge import merge_rooms
from packages.views.models import (
    OBJECT_TOLERANCE_M,
    MutableView,
    coerce_event,
    make_divider,
    payload_point_pair,
    polygon_from_room,
    recompute_neighbors,
)
from packages.views.restore import restore_face
from packages.views.split import split_room


def reduce_view(physical_layout: LayoutDto, events: Sequence[Any]) -> LayoutDto:
    physical_rooms = [room.model_copy(deep=True) for room in physical_layout.physical_rooms]
    view = MutableView(rooms={room.id: room.model_copy(deep=True) for room in physical_rooms})
    parsed = [coerce_event(item) for item in events]
    layout_rev = 0
    for event in parsed:
        layout_rev = max(layout_rev, event.layout_rev)
        if event.retracted_by_event_id:
            continue
        action = event.action_value
        if action == FeedbackAction.UNDO.value:
            continue
        if action == FeedbackAction.DELETE_ROOM.value:
            delete_room(view, event.target_ids)
        elif action == FeedbackAction.RESTORE_FACE.value:
            restore_face(
                view,
                event.target_ids,
                physical_rooms=physical_rooms,
                payload=dict(event.payload),
            )
        elif action == FeedbackAction.MERGE_ROOMS.value:
            merge_rooms(view, event.target_ids)
        elif action == FeedbackAction.SPLIT.value:
            pair = payload_point_pair(event.payload)
            if pair is not None:
                split_room(view, pair[0], pair[1], event.target_ids or None)
        elif action in {FeedbackAction.DRAW_WALL.value, FeedbackAction.CONNECT_NODES.value}:
            pair = payload_point_pair(event.payload)
            if pair is not None:
                divider = make_divider(pair[0], pair[1])
                view.dividers[divider.id] = divider
        elif action == FeedbackAction.RENAME_ROOM.value:
            name = event.payload.get("name")
            if isinstance(name, str) and name.strip():
                cleaned = name.strip()
                for room_id in event.target_ids:
                    room = view.rooms.get(room_id)
                    if room is None:
                        continue
                    dumped = room.model_dump()
                    dumped["name"] = cleaned
                    view.rooms[room_id] = RoomDto.model_validate(dumped)
                    view.names[room_id] = cleaned
        elif action in {FeedbackAction.ACCEPT.value, FeedbackAction.MARK_ENTITY.value}:
            continue

    rooms = recompute_neighbors(list(view.rooms.values()))
    rooms = _apply_names(rooms, view.names)
    objects, unassigned = _reassign_objects(physical_layout, rooms)
    texts = _reassign_texts(physical_layout, rooms)
    dumped = physical_layout.model_dump()
    dumped["physical_rooms"] = [room.model_dump() for room in physical_rooms]
    dumped["rooms"] = [room.model_dump() for room in rooms]
    dumped["virtual_dividers"] = [d.model_dump() for d in view.dividers.values()]
    dumped["objects"] = [o.model_dump() for o in objects]
    dumped["unassigned_objects"] = [o.model_dump() for o in unassigned]
    dumped["texts"] = [t.model_dump() for t in texts]
    dumped["meta"]["layout_rev"] = layout_rev
    return LayoutDto.model_validate(dumped)


def _apply_names(rooms: list[RoomDto], names: dict[str, str]) -> list[RoomDto]:
    if not names:
        return rooms
    out: list[RoomDto] = []
    for room in rooms:
        if room.id in names:
            dumped = room.model_dump()
            dumped["name"] = names[room.id]
            out.append(RoomDto.model_validate(dumped))
        else:
            out.append(room)
    return out


def _containing_room_id(rooms: list[RoomDto], x: float, y: float) -> str | None:
    point = Point(x, y)
    for room in rooms:
        geom = polygon_from_room(room)
        if geom.contains(point) or geom.buffer(OBJECT_TOLERANCE_M).contains(point):
            return room.id
    return None


def _reassign_objects(
    physical_layout: LayoutDto,
    rooms: list[RoomDto],
) -> tuple[list[ObjectDto], list[ObjectDto]]:
    combined = [
        *[o.model_copy(deep=True) for o in physical_layout.objects],
        *[o.model_copy(deep=True) for o in physical_layout.unassigned_objects],
    ]
    assigned: list[ObjectDto] = []
    unassigned: list[ObjectDto] = []
    room_objects: dict[str, list[str]] = {room.id: [] for room in rooms}
    for obj in combined:
        room_id = _containing_room_id(rooms, obj.center.x, obj.center.y)
        dumped = obj.model_dump()
        dumped["room_id"] = room_id
        updated = ObjectDto.model_validate(dumped)
        if room_id is None:
            unassigned.append(updated)
        else:
            assigned.append(updated)
            room_objects[room_id].append(updated.id)
    relocated: list[RoomDto] = []
    for room in rooms:
        dumped = room.model_dump()
        dumped["object_ids"] = room_objects.get(room.id, [])
        relocated.append(RoomDto.model_validate(dumped))
    rooms[:] = relocated
    return assigned, unassigned


def _reassign_texts(physical_layout: LayoutDto, rooms: list[RoomDto]) -> list[TextDto]:
    out: list[TextDto] = []
    for text in physical_layout.texts:
        dumped = text.model_dump()
        dumped["room_id"] = _containing_room_id(rooms, text.position.x, text.position.y)
        out.append(TextDto.model_validate(dumped))
    return out
