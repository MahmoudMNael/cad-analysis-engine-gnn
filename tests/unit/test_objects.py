from packages.assemble.objects import assemble_objects
from packages.assemble.rooms import PhysicalRoom
from packages.common.configuration import Settings
from packages.contracts.enums import DecisionSource, EntityClass, PrimitiveKind
from packages.ingest.models import primitive_from_parts
from packages.models_nn.common import EntityScore
from shapely.geometry import LineString, Polygon


def _score(pid: str) -> EntityScore:
    probs = {c.value: 0.01 for c in EntityClass}
    probs[EntityClass.FURNITURE.value] = 0.9
    return EntityScore(
        target_id=pid,
        probabilities=probs,
        predicted_class=EntityClass.FURNITURE.value,
        confidence=0.9,
        entropy=0.1,
        model_version="test",
        decision_source=DecisionSource.HEURISTIC,
        explanation="test",
    )


def _line(pid: str, a: tuple[float, float], b: tuple[float, float], block: str) -> object:
    return primitive_from_parts(
        primitive_id=pid,
        kind=PrimitiveKind.LINE,
        geometry=LineString([a, b]),
        layer="0",
        color=7,
        linetype="CONTINUOUS",
        source_entity_type="LINE",
        source_handle=pid,
        block_name=block,
        attributes={},
    )


def test_same_block_nearby_parts_cluster_to_one_object():
    room = PhysicalRoom(
        id="r1",
        face_id="r1",
        geometry=Polygon([(0, 0), (5, 0), (5, 4), (0, 4)]),
        area_m2=20.0,
        perimeter_m=18.0,
        center=(2.5, 2.0),
        bbox=(0.0, 0.0, 5.0, 4.0),
        confidence=1.0,
        kind="indoor",
        neighbor_ids=(),
        door_ids=(),
        window_ids=(),
        object_ids=(),
    )
    parts = (
        _line("a", (1.0, 1.0), (1.4, 1.0), "CHAIR-1"),
        _line("b", (1.0, 1.0), (1.0, 1.4), "CHAIR-1"),
        _line("c", (3.5, 2.0), (3.9, 2.0), "CHAIR-1"),
    )
    scores = {p.id: _score(p.id) for p in parts}
    assigned, unassigned, rooms = assemble_objects(parts, scores, (room,), Settings())
    placed = assigned + unassigned
    assert len(placed) == 2
    assert all(obj.room_id == "r1" for obj in placed)
    assert rooms[0].object_ids
