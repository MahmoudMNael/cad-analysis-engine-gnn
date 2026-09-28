from packages.common.configuration import Settings
from packages.ingest.models import primitive_from_parts
from packages.contracts.enums import GapDecision, PrimitiveKind
from packages.models_nn.heuristics import HeuristicEntityClassifier, HeuristicFaceClassifier, HeuristicGapClassifier
from packages.primitives.features import EntityFeatures
from shapely.geometry import LineString


def test_entity_heuristic_returns_full_probability_vector():
    prim = primitive_from_parts(
        primitive_id="p1",
        kind=PrimitiveKind.LINE,
        geometry=LineString([(0, 0), (3, 0)]),
        layer="WALL",
        color=7,
        linetype="CONTINUOUS",
        source_entity_type="LINE",
        source_handle="1A",
        block_name=None,
        attributes={},
    )
    features = EntityFeatures(primitive_id="p1", values={
        "length": 3.0, "area": 0.0, "bbox_width": 3.0, "bbox_height": 0.0,
        "orientation_sin": 0.0, "orientation_cos": 1.0, "aspect_ratio": 3.0,
        "endpoint_distance": 3.0, "curvature": 0.0,
    })
    score = HeuristicEntityClassifier().predict(prim, features, Settings())
    assert set(score.probabilities) >= {"wall", "opening", "furniture", "hatch", "dim", "text", "unknown"}
    assert abs(sum(score.probabilities.values()) - 1.0) < 1e-6
    assert score.predicted_class == "wall"


def test_face_heuristic_same_shape_as_gnn():
    features = {
        "area": 12.0,
        "perimeter": 14.0,
        "compactness": 0.7,
        "aspect_ratio": 1.2,
        "number_of_doors": 1.0,
        "adjacent_unbounded": 1.0,
        "text_overlap": 0.0,
    }
    score = HeuristicFaceClassifier().predict("f1", features, Settings())
    assert score.face_id == "f1"
    assert "indoor" in score.probabilities
    assert score.decision_source.value == "heuristic"
    assert score.explanation


def test_face_heuristic_cavity_from_area():
    score = HeuristicFaceClassifier().predict(
        "c1",
        {
            "area": 0.3,
            "perimeter": 2.4,
            "compactness": 0.6,
            "aspect_ratio": 1.0,
            "number_of_doors": 0.0,
            "adjacent_unbounded": 0.0,
            "text_overlap": 0.0,
            "wall_overlap": 0.2,
        },
        Settings(),
    )
    assert score.predicted_class == "cavity"
    assert score.decision_source.value == "heuristic"


def _features(**values: float) -> EntityFeatures:
    base = {
        "length": 1.0,
        "area": 0.0,
        "bbox_width": 1.0,
        "bbox_height": 0.0,
        "orientation_sin": 0.0,
        "orientation_cos": 1.0,
        "aspect_ratio": 1.0,
        "endpoint_distance": 1.0,
        "curvature": 0.0,
    }
    base.update(values)
    return EntityFeatures(primitive_id="p", values=base)


def test_entity_heuristic_door_arc_is_opening():
    prim = primitive_from_parts(
        primitive_id="arc1",
        kind=PrimitiveKind.ARC,
        geometry=LineString([(0.9, 0.0), (0.64, 0.64), (0.0, 0.9)]),
        layer="0",
        color=7,
        linetype="CONTINUOUS",
        source_entity_type="ARC",
        source_handle="A1",
        block_name="DOOR-01",
        attributes={"center": (0.0, 0.0), "radius": 0.9},
    )
    score = HeuristicEntityClassifier().predict(
        prim,
        _features(length=1.41, curvature=0.10, bbox_width=0.9, bbox_height=0.9, endpoint_distance=1.27),
        Settings(),
    )
    assert score.predicted_class == "opening"


def test_entity_heuristic_short_block_fixture_is_furniture():
    prim = primitive_from_parts(
        primitive_id="fx1",
        kind=PrimitiveKind.LINE,
        geometry=LineString([(0.0, 0.0), (0.4, 0.0)]),
        layer="0",
        color=7,
        linetype="CONTINUOUS",
        source_entity_type="LINE",
        source_handle="B1",
        block_name="WC-1",
        attributes={},
    )
    score = HeuristicEntityClassifier().predict(prim, _features(length=0.4, endpoint_distance=0.4), Settings())
    assert score.predicted_class == "furniture"


def test_entity_heuristic_plain_wall_line_stays_wall():
    prim = primitive_from_parts(
        primitive_id="w1",
        kind=PrimitiveKind.LINE,
        geometry=LineString([(0.0, 0.0), (4.0, 0.0)]),
        layer="0",
        color=7,
        linetype="CONTINUOUS",
        source_entity_type="LINE",
        source_handle="W1",
        block_name=None,
        attributes={},
    )
    score = HeuristicEntityClassifier().predict(prim, _features(length=4.0, endpoint_distance=4.0, bbox_width=4.0), Settings())
    assert score.predicted_class == "wall"


def test_gap_heuristic_rejects_non_collinear_interior_span():
    score = HeuristicGapClassifier().predict(
        "g1",
        0.8,
        collinear=False,
        crosses=False,
        already_connected=False,
        settings=Settings(),
        opening_like=False,
    )
    assert score.decision is GapDecision.REJECT


def test_gap_heuristic_still_portals_door_width_collinear():
    score = HeuristicGapClassifier().predict(
        "g2",
        0.9,
        collinear=True,
        crosses=False,
        already_connected=False,
        settings=Settings(),
        opening_like=True,
    )
    assert score.decision is GapDecision.PORTAL

