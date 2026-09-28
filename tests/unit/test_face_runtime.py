from packages.common.configuration import Settings
from packages.models_nn.common import FACE_CLASSES
from packages.models_nn.runtime import FaceRuntime


class _FakeGnn:
    def __init__(self, indoor: float) -> None:
        rest = (1.0 - indoor) / 4.0
        self.probs = {
            "indoor": indoor,
            "corridor": rest,
            "shaft": rest,
            "outdoor": rest,
            "cavity": rest,
        }

    def predict_proba(self, node_features, edge_index, edge_attr):
        return [dict(self.probs) for _ in node_features]


def test_heuristic_runtime_without_torch():
    runtime = FaceRuntime("heuristic:face_v1")
    score = runtime.predict(
        "f1",
        {
            "area": 12.0,
            "perimeter": 14.0,
            "compactness": 0.7,
            "aspect_ratio": 1.2,
            "number_of_doors": 1.0,
            "adjacent_unbounded": 0.0,
            "text_overlap": 1.0,
        },
        Settings(),
    )
    assert score.decision_source.value == "heuristic"
    assert set(score.probabilities) >= set(FACE_CLASSES)
    assert runtime.decision_source_name == "heuristic"


def test_missing_gnn_checkpoint_falls_back_to_heuristic():
    runtime = FaceRuntime("gnn:face:missing", checkpoint_uri=None)
    score = runtime.predict("f1", {"area": 12.0, "perimeter": 14.0, "compactness": 0.7}, Settings())
    assert runtime._gnn is None
    assert score.decision_source.value == "heuristic"


def test_confident_gnn_uses_model_decision_source():
    runtime = FaceRuntime("gnn:face:test")
    runtime._gnn = _FakeGnn(0.9)
    score = runtime.predict("f1", {"area": 12.0}, Settings(face_threshold=0.35))
    assert score.predicted_class == "indoor"
    assert score.decision_source.value == "model"
    assert score.model_version == "gnn:face:test"


def test_uncertain_gnn_falls_back_to_heuristic():
    runtime = FaceRuntime("gnn:face:test")
    runtime._gnn = _FakeGnn(0.2)
    score = runtime.predict(
        "f1",
        {
            "area": 12.0,
            "perimeter": 14.0,
            "compactness": 0.7,
            "aspect_ratio": 1.2,
            "number_of_doors": 1.0,
            "adjacent_unbounded": 0.0,
            "text_overlap": 1.0,
        },
        Settings(face_threshold=0.35),
    )
    assert score.decision_source.value == "heuristic"
    assert "gnn uncertain" in score.explanation
    assert score.model_version == "gnn:face:test"
