from __future__ import annotations

import json

from packages.common.configuration import Settings
from packages.contracts.enums import LabelSource
from packages.eval.datasets import TrainingRow, face_graphs_from_rows
from packages.eval.face_bundle import GRAPH_JSON_KEYS, build_face_dataset_payload
from packages.models_nn.face_train_kernel import FACE_NODE_FEATURE_KEYS as KERNEL_NODE_KEYS
from packages.models_nn.features.face import FACE_NODE_FEATURE_KEYS


def _row(
    *,
    job_id: str,
    target_id: str,
    label: str,
    source: LabelSource = LabelSource.USER_EXPLICIT,
    weight: float = 0.75,
) -> TrainingRow:
    return TrainingRow(
        sample_id=f"{job_id}-{target_id}-{label}",
        job_id=job_id,
        building_id=job_id,
        target_id=target_id,
        feature_schema_version="face_features_v1",
        features={"area": 12.0, "perimeter": 14.0},
        label=label,
        label_source=source,
        weight=weight,
        engine_sha="sha",
        model_input_version="face_features_v1",
    )


def test_kernel_constants_match_repo_schema():
    assert KERNEL_NODE_KEYS == FACE_NODE_FEATURE_KEYS


def test_export_without_labels_is_not_eligible():
    settings = Settings()
    payload = build_face_dataset_payload(rows=[], graphs=[], settings=settings, engine_sha="engine")
    assert payload["eligibility"]["ok"] is False
    assert payload["graphs"] == []
    assert payload["eligibility"]["labeled_jobs"] == 0
    assert payload["feature_schema_version"] == "face_features_v1"


def test_export_graphs_are_split_ready_without_cad_payloads():
    rows = [
        _row(job_id="job-a", target_id="f1", label="indoor"),
        _row(job_id="job-a", target_id="f2", label="cavity"),
        _row(job_id="job-b", target_id="f1", label="indoor"),
    ]
    graphs = face_graphs_from_rows(rows)
    payload = build_face_dataset_payload(
        rows=rows,
        graphs=graphs,
        settings=Settings(face_train_min_labeled_jobs=2, face_train_min_classes=2),
        engine_sha="engine",
    )
    dumped = json.dumps(payload)
    assert "wkt" not in dumped
    assert "dxf" not in dumped.lower()
    assert payload["eligibility"]["ok"] is True
    assert payload["graphs"]
    for graph in payload["graphs"]:
        assert tuple(graph.keys()) == GRAPH_JSON_KEYS
        assert graph["job_id"]
        assert graph["building_id"]
        assert graph["face_ids"]
        assert graph["node_features"]
        assert graph["labels"]
        assert graph["weights"]
    assert "hyperparams" in payload
    assert payload["hyperparams"]["face_gnn_epochs"] >= 1
    assert payload["classes"] == ["indoor", "corridor", "shaft", "outdoor", "cavity"]
