from __future__ import annotations

from collections.abc import Sequence

from packages.contracts.dto.layout import RoomDto
from packages.views.models import MutableView


def matching_room_ids(rooms: dict[str, RoomDto], target_ids: Sequence[str]) -> list[str]:
    wanted = set(target_ids)
    matched: list[str] = []
    for room in rooms.values():
        if room.id in wanted or (room.physical_room_id and room.physical_room_id in wanted):
            matched.append(room.id)
    return matched


def delete_room(view: MutableView, target_ids: Sequence[str]) -> list[str]:
    removed = matching_room_ids(view.rooms, target_ids)
    for room_id in removed:
        view.rooms.pop(room_id, None)
    return removed
