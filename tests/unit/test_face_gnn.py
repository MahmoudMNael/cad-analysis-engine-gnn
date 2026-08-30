from pathlib import Path

import pytest

from packages.common.configuration import Settings
from packages.eval.datasets import FaceGraphSample
from packages.models_nn.common import FACE_CLASSES
from packages.models_nn.features.face import FACE_NODE_FEATURE_KEYS

torch = pytest.importorskip("torch")


def _graph(job_id: str, label: int, area: float, building: str | None = None) -> FaceGraphSample:
    features = [0.0] * len(FACE_NODE_FEATURE_KEYS)
    features[0] = area
    features[1] = max(area**0.5 * 4.0, 1.0)
    return FaceGraphSample(
        job_id=job_id,
        building_id=building or job_id,
        face_ids=[f"{job_id}-f0"],
        node_features=[features],
        edge_index=[],
        edge_attr=[],
        labels=[label],
        weights=[1.0],
    )


def test_face_gnn_forward_and_checkpoint(tmp_path: Path):
    from packages.models_nn.face_gnn import FaceGNN, pack_pyg_data, pyg_available

    model = FaceGNN(hidden_dim=8, num_layers=1, dropout=0.0)
    graphs = [_graph("a", 0, 20.0), _graph("b", 4, 0.3)]
    packed = pack_pyg_data(graphs[0].node_features, graphs[0].edge_index, graphs[0].edge_attr, graphs[0].labels)
    if pyg_available():
        assert packed.x.shape[0] == 1
    proba = model.predict_proba(graphs[0].node_features, [], [])
    assert set(proba[0]) == set(FACE_CLASSES)
    assert abs(sum(proba[0].values()) - 1.0) < 1e-5
    path = tmp_path / "face.pt"
    model.save(path)
    loaded = FaceGNN.load(path)
    again = loaded.predict_proba(graphs[0].node_features, [], [])
    assert again[0].keys() == proba[0].keys()


def test_training_loop_runs_on_synthetic_graphs():
    from apps.trainer.evaluate import evaluate_graphs
    from apps.trainer.train import train_face_on_graphs

    indoor = [_graph(f"in-{i}", 0, 18.0 + i, building=f"b{i}") for i in range(4)]
    cavity = [_graph(f"cav-{i}", 4, 0.2 + i * 0.01, building=f"c{i}") for i in range(4)]
    settings = Settings(face_gnn_epochs=5, face_gnn_hidden_dim=16, face_gnn_layers=1, face_gnn_dropout=0.0)
    model, info = train_face_on_graphs(indoor + cavity, indoor[:1] + cavity[:1], settings=settings, engine_sha="test")
    metrics = evaluate_graphs(model, indoor + cavity)
    assert info["best_val_macro_f1"] >= 0.0
    assert metrics["n"] == 8
    assert 0.0 <= metrics["macro_f1"] <= 1.0
