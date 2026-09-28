from packages.common.configuration import Settings
from packages.contracts.enums import DecisionSource, EntityClass, PrimitiveKind
from packages.graphkit.graph_builder import wall_candidate_primitives
from packages.ingest.models import primitive_from_parts
from packages.models_nn.common import EntityScore
from shapely.geometry import LineString


def _score(prim_id: str, predicted: str, wall_p: float = 0.1) -> EntityScore:
    probs = {c.value: 0.01 for c in EntityClass}
    probs[predicted] = max(wall_p, 0.8) if predicted != EntityClass.WALL.value else 0.9
    probs[EntityClass.WALL.value] = wall_p
    return EntityScore(
        target_id=prim_id,
        probabilities=probs,
        predicted_class=predicted,
        confidence=0.9,
        entropy=0.1,
        model_version="test",
        decision_source=DecisionSource.HEURISTIC,
        explanation="test",
    )


def _line(pid: str, layer: str = "0", kind: PrimitiveKind = PrimitiveKind.LINE, block: str | None = None):
    return primitive_from_parts(
        primitive_id=pid,
        kind=kind,
        geometry=LineString([(0.0, 0.0), (3.0, 0.0)]),
        layer=layer,
        color=7,
        linetype="CONTINUOUS",
        source_entity_type=kind.value.upper(),
        source_handle=pid,
        block_name=block,
        attributes={},
    )


def test_wall_candidates_skip_openings_and_furniture():
    wall = _line("w")
    door = _line("d", layer="DOOR", kind=PrimitiveKind.ARC)
    chair = _line("c", layer="FURNITURE")
    scores = {
        "w": _score("w", EntityClass.WALL.value, wall_p=0.9),
        "d": _score("d", EntityClass.OPENING.value, wall_p=0.2),
        "c": _score("c", EntityClass.FURNITURE.value, wall_p=0.2),
    }
    out = wall_candidate_primitives((wall, door, chair), scores, Settings())
    assert [p.id for p in out] == ["w"]


def test_wall_candidates_skip_unlabeled_door_arc():
    arc = primitive_from_parts(
        primitive_id="a",
        kind=PrimitiveKind.ARC,
        geometry=LineString([(0.9, 0.0), (0.64, 0.64), (0.0, 0.9)]),
        layer="0",
        color=7,
        linetype="CONTINUOUS",
        source_entity_type="ARC",
        source_handle="a",
        block_name=None,
        attributes={},
    )
    scores = {"a": _score("a", EntityClass.WALL.value, wall_p=0.9)}
    out = wall_candidate_primitives((arc,), scores, Settings())
    assert out == []
