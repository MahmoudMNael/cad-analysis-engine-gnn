from packages.common.configuration import Settings
from packages.contracts.enums import PrimitiveKind
from packages.ingest.models import Primitive, primitive_from_parts
from packages.models_nn.heuristics import HeuristicEntityClassifier
from packages.models_nn.runtime import EntityRuntime
from packages.primitives.features import NEIGHBOR_FEATURE_KEYS, extract_entity_features
from packages.primitives.spatial_index import build_primitive_indexes
from shapely.geometry import LineString


def _wall_line(primitive_id: str, x: float) -> Primitive:
    return primitive_from_parts(
        primitive_id=primitive_id,
        kind=PrimitiveKind.LINE,
        geometry=LineString([(x, 0.0), (x, 2.0)]),
        layer="WALL",
        color=7,
        linetype="CONTINUOUS",
        source_entity_type="LINE",
        source_handle=primitive_id,
        block_name=None,
        attributes={},
    )


def _cluster():
    prims = tuple(_wall_line(name, offset) for name, offset in (("a", 0.0), ("b", 0.02), ("c", 0.04)))
    return prims, {p.id: p for p in prims}, build_primitive_indexes(prims)


def test_entity_runtime_does_not_require_neighbors():
    assert EntityRuntime("heuristic:entity_v1").requires_neighbor_features is False
    assert EntityRuntime("gnn:entity_unused").requires_neighbor_features is False


def test_heuristic_scores_match_with_and_without_neighbors():
    prims, by_id, indexes = _cluster()
    classifier = HeuristicEntityClassifier()
    settings = Settings()
    for prim in prims:
        with_n = extract_entity_features(prim, by_id, indexes, include_neighbors=True)
        without = extract_entity_features(prim, by_id, indexes, include_neighbors=False)
        score_a = classifier.predict(prim, with_n, settings)
        score_b = classifier.predict(prim, without, settings)
        assert score_a.probabilities == score_b.probabilities
        assert score_a.predicted_class == score_b.predicted_class
        assert score_a.confidence == score_b.confidence
        local_keys = [k for k in with_n.values if k not in NEIGHBOR_FEATURE_KEYS]
        assert {k: with_n.values[k] for k in local_keys} == {k: without.values[k] for k in local_keys}


def test_neighbor_keys_zero_when_skipped():
    prims, by_id, indexes = _cluster()
    prim = prims[1]
    with_n = extract_entity_features(prim, by_id, indexes, include_neighbors=True)
    without = extract_entity_features(prim, by_id, indexes, include_neighbors=False)
    assert set(NEIGHBOR_FEATURE_KEYS) <= set(with_n.values)
    assert set(NEIGHBOR_FEATURE_KEYS) <= set(without.values)
    assert all(without.values[k] == 0.0 for k in NEIGHBOR_FEATURE_KEYS)
    assert with_n.values["neighbor_count"] >= 2.0
    assert with_n.values["local_density"] == with_n.values["neighbor_count"]
    assert with_n.values["endpoint_degree"] >= 1.0
