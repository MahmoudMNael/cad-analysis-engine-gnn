from __future__ import annotations

from collections.abc import Sequence

from packages.contracts.dto.layout import FaceDto, RoomDto
from packages.views.models import MutableView, polygon_from_face, room_from_polygon


def restore_face(
    view: MutableView,
    target_ids: Sequence[str],
    *,
    physical_rooms: Sequence[RoomDto],
    payload: dict,
    faces: Sequence[FaceDto] = (),
) -> list[str]:
    physical_by_id = {room.id: room for room in physical_rooms}
    faces_by_id = {face.id: face for face in faces}
    visible = {room.id for room in view.rooms.values()}
    visible.update(room.physical_room_id for room in view.rooms.values() if room.physical_room_id)
    restored: list[str] = []
    for face_id in target_ids:
        if face_id in visible:
            continue
        if face_id in physical_by_id:
            room = physical_by_id[face_id].model_copy(deep=True)
            view.rooms[room.id] = room
            visible.add(room.id)
            restored.append(room.id)
            continue
        face = faces_by_id.get(face_id)
        if face is not None:
            room = room_from_polygon(
                polygon_from_face(face),
                room_id=face.id,
                kind="indoor",
                confidence=face.confidence,
                physical_room_id=face.id,
                parent_room_id=None,
            )
            view.rooms[room.id] = room
            visible.add(room.id)
            restored.append(room.id)
            continue
        boundary = payload.get("boundary")
        if not boundary:
            continue
        from shapely.geometry import Polygon

        pts = [(float(p["x"]), float(p["y"])) for p in boundary]
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        room = room_from_polygon(
            Polygon(pts),
            room_id=face_id,
            kind=str(payload.get("kind") or "indoor"),
            confidence=float(payload.get("confidence") or 1.0),
            physical_room_id=face_id,
            parent_room_id=None,
            name=payload.get("name"),
        )
        view.rooms[room.id] = room
        visible.add(room.id)
        restored.append(room.id)
    return restored
