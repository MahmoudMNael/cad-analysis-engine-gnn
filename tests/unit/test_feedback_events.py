from pydantic import ValidationError

from packages.common.errors import InvalidTargetError
from packages.contracts.dto.feedback import DividerRequest, FeedbackEventRequest
from packages.contracts.enums import FeedbackAction, EntityClass
from packages.views.validate import prepare_feedback
from tests.unit.test_view_reducer import _event, _layout, _room


def test_feedback_request_forbids_extra_fields():
    try:
        FeedbackEventRequest(action=FeedbackAction.ACCEPT, extra_field=True)  # type: ignore[call-arg]
        raise AssertionError("expected validation error")
    except ValidationError:
        pass


def test_divider_rejects_non_finite_coordinates():
    try:
        DividerRequest(start_x=float("nan"), start_y=0, end_x=1, end_y=1)
        raise AssertionError("expected validation error")
    except ValidationError:
        pass


def test_prepare_delete_unknown_room():
    layout = _layout(_room("r1", 0, 0, 5, 4))
    try:
        prepare_feedback(
            action=FeedbackAction.DELETE_ROOM,
            target_ids=["missing"],
            payload={},
            actor="user",
            layout=layout,
            events=[],
        )
        raise AssertionError("expected invalid target")
    except InvalidTargetError:
        pass


def test_prepare_mark_entity_requires_class():
    layout = _layout(_room("r1", 0, 0, 5, 4))
    try:
        prepare_feedback(
            action=FeedbackAction.MARK_ENTITY,
            target_ids=["prim-1"],
            payload={},
            actor="user",
            layout=layout,
            events=[],
        )
        raise AssertionError("expected invalid target")
    except InvalidTargetError:
        pass
    prepared = prepare_feedback(
        action=FeedbackAction.MARK_ENTITY,
        target_ids=["prim-1"],
        payload={"class": EntityClass.WALL.value},
        actor="expert",
        layout=layout,
        events=[],
    )
    assert prepared.payload["class"] == "wall"


def test_prepare_split_requires_partition():
    layout = _layout(_room("r1", 0, 0, 5, 4))
    try:
        prepare_feedback(
            action=FeedbackAction.SPLIT,
            target_ids=["r1"],
            payload={"start_x": 100.0, "start_y": 100.0, "end_x": 101.0, "end_y": 101.0},
            actor="user",
            layout=layout,
            events=[],
        )
        raise AssertionError("expected invalid target")
    except InvalidTargetError:
        pass
    prepared = prepare_feedback(
        action=FeedbackAction.SPLIT,
        target_ids=["r1"],
        payload={"start_x": 2.5, "start_y": -1.0, "end_x": 2.5, "end_y": 5.0},
        actor="user",
        layout=layout,
        events=[],
    )
    assert prepared.target_ids == ["r1"]


def test_prepare_split_without_targets_scopes_to_the_stroked_room():
    layout = _layout(_room("left", 0, 0, 5, 4), _room("right", 5, 0, 10, 4))
    prepared = prepare_feedback(
        action=FeedbackAction.SPLIT,
        target_ids=[],
        payload={"start_x": 2.5, "start_y": 0.0, "end_x": 2.5, "end_y": 4.0},
        actor="user",
        layout=layout,
        events=[],
    )
    assert prepared.target_ids == ["left"]


def test_prepare_undo_without_history():
    from packages.common.errors import InvalidFeedbackActionError

    layout = _layout(_room("r1", 0, 0, 5, 4))
    try:
        prepare_feedback(
            action=FeedbackAction.UNDO,
            target_ids=[],
            payload={},
            actor="user",
            layout=layout,
            events=[],
        )
        raise AssertionError("expected invalid action")
    except InvalidFeedbackActionError:
        pass


def test_prepare_undo_already_retracted():
    from packages.common.errors import EventAlreadyRetractedError

    layout = _layout(_room("r1", 0, 0, 5, 4))
    event = _event(FeedbackAction.DELETE_ROOM, target_ids=["r1"], sequence=1, retracted="evt-2")
    try:
        prepare_feedback(
            action=FeedbackAction.UNDO,
            target_ids=["evt-1"],
            payload={},
            actor="user",
            layout=layout,
            events=[event],
        )
        raise AssertionError("expected already retracted")
    except EventAlreadyRetractedError:
        pass
