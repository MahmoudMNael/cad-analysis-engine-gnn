from packages.contracts.dto.common import BBoxDto, PointDto
from packages.contracts.dto.layout import LayoutDto, RoomDto
from packages.contracts.dto.feedback import FeedbackEventDto
from packages.contracts.enums import FeedbackAction
from packages.views.reducer import reduce_view
from tests.unit.test_dto_validation import _meta


def _room(room_id: str, minx: float, miny: float, maxx: float, maxy: float, kind: str = "indoor") -> RoomDto:
    return RoomDto(
        id=room_id,
        boundary=[
            PointDto(x=minx, y=miny),
            PointDto(x=maxx, y=miny),
            PointDto(x=maxx, y=maxy),
            PointDto(x=minx, y=maxy),
        ],
        area_m2=(maxx - minx) * (maxy - miny),
        area_net_m2=(maxx - minx) * (maxy - miny),
        area_gross_m2=(maxx - minx) * (maxy - miny),
        center=PointDto(x=(minx + maxx) / 2, y=(miny + maxy) / 2),
        perimeter_m=2 * ((maxx - minx) + (maxy - miny)),
        bbox=BBoxDto(min_x=minx, min_y=miny, max_x=maxx, max_y=maxy),
        confidence=0.9,
        kind=kind,
        physical_room_id=room_id,
    )


def _layout(*rooms: RoomDto) -> LayoutDto:
    physical = [room.model_copy(deep=True) for room in rooms]
    return LayoutDto(
        unit="m",
        meta=_meta(),
        rooms=[room.model_copy(deep=True) for room in physical],
        physical_rooms=physical,
    )


def _event(action: FeedbackAction, *, target_ids: list[str], payload: dict | None = None, sequence: int = 1, retracted: str | None = None) -> FeedbackEventDto:
    return FeedbackEventDto(
        event_id=f"evt-{sequence}",
        job_id="job1",
        sequence=sequence,
        ts="2026-01-01T00:00:00+00:00",
        actor="user",
        action=action,
        target_ids=target_ids,
        payload=payload or {},
        layout_rev=sequence,
        retracted_by_event_id=retracted,
    )


def test_reducer_does_not_mutate_input():
    layout = _layout(_room("r1", 0, 0, 5, 4))
    original = layout.model_dump()
    reduce_view(layout, [_event(FeedbackAction.DELETE_ROOM, target_ids=["r1"])])
    assert layout.model_dump() == original


def test_initial_view_copies_physical_rooms():
    layout = LayoutDto(unit="m", meta=_meta(), physical_rooms=[], rooms=[])
    derived = reduce_view(layout, ())
    assert derived.rooms == derived.physical_rooms
    assert derived.meta.layout_rev == 0


def test_delete_room_removes_derived_only():
    layout = _layout(_room("r1", 0, 0, 5, 4), _room("r2", 5, 0, 10, 4))
    derived = reduce_view(layout, [_event(FeedbackAction.DELETE_ROOM, target_ids=["r1"])])
    assert [r.id for r in derived.physical_rooms] == ["r1", "r2"]
    assert [r.id for r in derived.rooms] == ["r2"]
    assert derived.meta.layout_rev == 1


def test_undo_event_is_skipped_when_retracted():
    layout = _layout(_room("r1", 0, 0, 5, 4))
    events = [
        _event(FeedbackAction.DELETE_ROOM, target_ids=["r1"], sequence=1, retracted="evt-2"),
        _event(FeedbackAction.UNDO, target_ids=["evt-1"], sequence=2),
    ]
    derived = reduce_view(layout, events)
    assert [r.id for r in derived.rooms] == ["r1"]
    assert [r.id for r in derived.physical_rooms] == ["r1"]


def test_split_creates_two_rooms_and_divider_without_changing_physical():
    layout = _layout(_room("r1", 0, 0, 5, 4))
    event = _event(
        FeedbackAction.SPLIT,
        target_ids=["r1"],
        payload={"start_x": 2.5, "start_y": -1.0, "end_x": 2.5, "end_y": 5.0},
    )
    derived = reduce_view(layout, [event])
    assert len(derived.physical_rooms) == 1
    assert derived.physical_rooms[0].id == "r1"
    assert len(derived.rooms) == 2
    assert abs(sum(r.area_m2 for r in derived.rooms) - 20.0) < 0.01
    assert len(derived.virtual_dividers) == 1
    again = reduce_view(layout, [event])
    assert [r.id for r in again.rooms] == [r.id for r in derived.rooms]


def test_merge_adjacent_rooms():
    layout = _layout(_room("r1", 0, 0, 5, 4), _room("r2", 5, 0, 10, 4))
    derived = reduce_view(layout, [_event(FeedbackAction.MERGE_ROOMS, target_ids=["r1", "r2"])])
    assert len(derived.physical_rooms) == 2
    assert len(derived.rooms) == 1
    assert abs(derived.rooms[0].area_m2 - 40.0) < 0.01


def test_rename_and_restore():
    layout = _layout(_room("r1", 0, 0, 5, 4))
    deleted = reduce_view(layout, [_event(FeedbackAction.DELETE_ROOM, target_ids=["r1"], sequence=1)])
    assert deleted.rooms == []
    restored = reduce_view(
        layout,
        [
            _event(FeedbackAction.DELETE_ROOM, target_ids=["r1"], sequence=1),
            _event(FeedbackAction.RESTORE_FACE, target_ids=["r1"], sequence=2),
        ],
    )
    assert [r.id for r in restored.rooms] == ["r1"]
    renamed = reduce_view(
        layout,
        [_event(FeedbackAction.RENAME_ROOM, target_ids=["r1"], payload={"name": "Kitchen"}, sequence=1)],
    )
    assert renamed.rooms[0].name == "Kitchen"
    assert renamed.physical_rooms[0].name is None


def test_split_without_targets_only_cuts_the_room_the_stroke_is_in():
    layout = _layout(_room("left", 0, 0, 5, 4), _room("right", 5, 0, 10, 4))
    derived = reduce_view(
        layout,
        [
            _event(
                FeedbackAction.SPLIT,
                target_ids=[],
                payload={"start_x": 2.5, "start_y": 0.0, "end_x": 2.5, "end_y": 4.0},
            )
        ],
    )
    assert len(derived.physical_rooms) == 2
    assert len(derived.rooms) == 3
    assert any(room.id == "right" for room in derived.rooms)
    assert all(room.id != "left" for room in derived.rooms)
    assert abs(sum(room.area_m2 for room in derived.rooms) - 40.0) < 0.01
    assert len(derived.virtual_dividers) == 1
    divider = derived.virtual_dividers[0]
    assert abs(divider.start.x - 2.5) < 1e-6
    assert abs(divider.end.x - 2.5) < 1e-6
    assert {round(divider.start.y, 6), round(divider.end.y, 6)} == {0.0, 4.0}


def test_split_ignores_other_rooms_on_the_same_axis():
    layout = _layout(_room("bottom", 0, 0, 5, 4), _room("top", 0, 4, 5, 8))
    derived = reduce_view(
        layout,
        [
            _event(
                FeedbackAction.SPLIT,
                target_ids=[],
                payload={"start_x": 2.5, "start_y": 0.0, "end_x": 2.5, "end_y": 4.0},
            )
        ],
    )
    assert len(derived.rooms) == 3
    assert any(room.id == "top" for room in derived.rooms)
    assert abs(sum(room.area_m2 for room in derived.rooms if room.id == "top") - 20.0) < 0.01


def test_draw_wall_adds_virtual_divider_only():
    layout = _layout(_room("r1", 0, 0, 5, 4))
    derived = reduce_view(
        layout,
        [
            _event(
                FeedbackAction.DRAW_WALL,
                target_ids=[],
                payload={"start_x": 0, "start_y": 0, "end_x": 1, "end_y": 0},
            )
        ],
    )
    assert len(derived.rooms) == 1
    assert len(derived.virtual_dividers) == 1
    assert derived.physical_rooms[0].id == "r1"
