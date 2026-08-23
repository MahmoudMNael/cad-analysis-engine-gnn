from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from packages.common.errors import (
    EventAlreadyRetractedError,
    InvalidFeedbackActionError,
    InvalidTargetError,
)
from packages.contracts.dto.feedback import FeedbackEventDto
from packages.contracts.dto.layout import FaceDto, LayoutDto
from packages.contracts.enums import EntityClass, FeedbackAction
from packages.views.models import MutableView, payload_point_pair
from packages.views.merge import merge_rooms
from packages.views.split import rooms_split_by_line


@dataclass
class PreparedFeedback:
    action: FeedbackAction
    target_ids: list[str]
    payload: dict[str, Any]
    actor: str
    retract_event_id: str | None = None
    clear_retraction_id: str | None = None


DIVIDER_ACTIONS = {
    FeedbackAction.SPLIT.value,
    FeedbackAction.DRAW_WALL.value,
    FeedbackAction.CONNECT_NODES.value,
}


def _room_index(layout: LayoutDto) -> dict[str, str]:
    index: dict[str, str] = {}
    for room in layout.rooms:
        index[room.id] = room.id
        if room.physical_room_id:
            index.setdefault(room.physical_room_id, room.id)
    return index


def _unretracted(events: list[FeedbackEventDto]) -> list[FeedbackEventDto]:
    return [event for event in events if event.retracted_by_event_id is None]


def prepare_feedback(
    *,
    action: FeedbackAction,
    target_ids: list[str],
    payload: dict[str, Any],
    actor: str,
    layout: LayoutDto,
    events: list[FeedbackEventDto],
    faces: list[FaceDto] | None = None,
    plan_graph: dict[str, Any] | None = None,
) -> PreparedFeedback:
    payload = dict(payload)
    targets = list(target_ids)
    if action == FeedbackAction.UNDO:
        return _prepare_undo(targets, events, actor)
    if action == FeedbackAction.DELETE_ROOM:
        index = _room_index(layout)
        if not targets:
            raise InvalidTargetError("delete_room requires target room ids")
        missing = [tid for tid in targets if tid not in index]
        if missing:
            raise InvalidTargetError("unknown room id", {"target_ids": ",".join(missing)})
        return PreparedFeedback(action, targets, payload, actor)
    if action == FeedbackAction.RESTORE_FACE:
        if not targets:
            raise InvalidTargetError("restore_face requires face ids")
        face_ids = {face.id for face in (faces or [])}
        physical_ids = {room.id for room in layout.physical_rooms}
        missing = [tid for tid in targets if tid not in face_ids and tid not in physical_ids]
        if missing:
            raise InvalidTargetError("unknown face id", {"target_ids": ",".join(missing)})
        visible = {room.id for room in layout.rooms} | {
            room.physical_room_id for room in layout.rooms if room.physical_room_id
        }
        if all(tid in visible for tid in targets):
            raise InvalidTargetError("face is already visible in the derived view")
        face_map = {face.id: face for face in (faces or [])}
        if targets and targets[0] in face_map:
            face = face_map[targets[0]]
            payload.setdefault(
                "boundary",
                [{"x": p.x, "y": p.y} for p in face.boundary],
            )
            payload.setdefault("confidence", face.confidence)
            payload.setdefault("kind", "indoor")
        return PreparedFeedback(action, targets, payload, actor)
    if action == FeedbackAction.MERGE_ROOMS:
        index = _room_index(layout)
        if len(targets) < 2:
            raise InvalidTargetError("merge_rooms requires at least two room ids")
        missing = [tid for tid in targets if tid not in index]
        if missing:
            raise InvalidTargetError("unknown room id", {"target_ids": ",".join(missing)})
        resolved = [index[tid] for tid in targets]
        probe = MutableView(rooms={room.id: room.model_copy(deep=True) for room in layout.rooms})
        merge_rooms(probe, resolved)
        return PreparedFeedback(action, resolved, payload, actor)
    if action == FeedbackAction.SPLIT:
        pair = payload_point_pair(payload)
        if pair is None:
            raise InvalidTargetError("split requires start and end coordinates")
        split_map = rooms_split_by_line(layout.rooms, pair[0], pair[1], targets or None)
        if not split_map:
            raise InvalidTargetError("divider does not partition a room")
        payload["start_x"], payload["start_y"] = pair[0]
        payload["end_x"], payload["end_y"] = pair[1]
        return PreparedFeedback(action, list(split_map.keys()), payload, actor)
    if action == FeedbackAction.DRAW_WALL:
        pair = payload_point_pair(payload)
        if pair is None:
            raise InvalidTargetError("draw_wall requires start and end coordinates")
        payload["start_x"], payload["start_y"] = pair[0]
        payload["end_x"], payload["end_y"] = pair[1]
        return PreparedFeedback(action, targets, payload, actor)
    if action == FeedbackAction.CONNECT_NODES:
        if len(targets) != 2:
            raise InvalidTargetError("connect_nodes requires exactly two node ids")
        nodes = {n["id"]: n for n in (plan_graph or {}).get("nodes", [])}
        missing = [tid for tid in targets if tid not in nodes]
        if missing:
            raise InvalidTargetError("unknown graph node", {"target_ids": ",".join(missing)})
        a = nodes[targets[0]]["point"]
        b = nodes[targets[1]]["point"]
        payload["node_a"] = targets[0]
        payload["node_b"] = targets[1]
        payload["node_a_point"] = list(a)
        payload["node_b_point"] = list(b)
        payload["start_x"], payload["start_y"] = float(a[0]), float(a[1])
        payload["end_x"], payload["end_y"] = float(b[0]), float(b[1])
        return PreparedFeedback(action, targets, payload, actor)
    if action == FeedbackAction.MARK_ENTITY:
        if not targets:
            raise InvalidTargetError("mark_entity requires a target id")
        label = payload.get("class") or payload.get("label") or payload.get("predicted_class")
        if not label:
            raise InvalidTargetError("mark_entity requires payload.class")
        try:
            EntityClass(str(label))
        except ValueError as exc:
            raise InvalidTargetError(f"invalid entity class: {label}") from exc
        payload["class"] = str(label)
        return PreparedFeedback(action, targets, payload, actor)
    if action == FeedbackAction.RENAME_ROOM:
        index = _room_index(layout)
        if len(targets) != 1:
            raise InvalidTargetError("rename_room requires a single room id")
        if targets[0] not in index:
            raise InvalidTargetError("unknown room id", {"target_ids": targets[0]})
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise InvalidTargetError("rename_room requires payload.name")
        payload["name"] = name.strip()
        return PreparedFeedback(action, [index[targets[0]]], payload, actor)
    if action == FeedbackAction.ACCEPT:
        payload["accepted_room_ids"] = [room.id for room in layout.rooms]
        return PreparedFeedback(action, [room.id for room in layout.rooms], payload, actor)
    raise InvalidFeedbackActionError(f"unsupported feedback action: {action}")


def _prepare_undo(
    target_ids: list[str],
    events: list[FeedbackEventDto],
    actor: str,
) -> PreparedFeedback:
    by_id = {event.event_id: event for event in events}
    if target_ids:
        target = by_id.get(target_ids[0])
        if target is None:
            raise InvalidTargetError("unknown event id", {"target_ids": target_ids[0]})
    else:
        unretracted = _unretracted(events)
        if not unretracted:
            raise InvalidFeedbackActionError("nothing to undo")
        target = unretracted[-1]
    if target.retracted_by_event_id:
        raise EventAlreadyRetractedError(target.event_id)
    clear_id = None
    if target.action == FeedbackAction.UNDO:
        for event in events:
            if event.retracted_by_event_id == target.event_id:
                clear_id = event.event_id
                break
    return PreparedFeedback(
        action=FeedbackAction.UNDO,
        target_ids=[target.event_id],
        payload={"retracted_event_id": target.event_id},
        actor=actor,
        retract_event_id=target.event_id,
        clear_retraction_id=clear_id,
    )


def divider_events_to_undo(events: list[FeedbackEventDto]) -> list[FeedbackEventDto]:
    unretracted = [
        event
        for event in _unretracted(events)
        if event.action.value in DIVIDER_ACTIONS
    ]
    return list(reversed(unretracted))
