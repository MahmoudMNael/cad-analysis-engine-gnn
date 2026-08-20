from packages.common.configuration import Settings
from packages.ingest.models import primitive_from_parts, BBox
from packages.contracts.enums import PrimitiveKind
from packages.models_nn.heuristics import HeuristicEntityClassifier, HeuristicFaceClassifier
from packages.primitives.features import EntityFeatures
from shapely.geometry import LineString, Polygon


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
