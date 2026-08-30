"""JSON face-graph export for the notebook trainer. No DXF, WKT, or snapshots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from packages.common.configuration import Settings
from packages.eval.datasets import FaceGraphSample, TrainingRow, validate_face_dataset
from packages.models_nn.common import FACE_CLASSES
from packages.models_nn.face_train_kernel import graph_to_dict
from packages.models_nn.features.face import (
    FACE_EDGE_FEATURE_KEYS,
    FACE_EDGE_FEATURE_SCHEMA_VERSION,
    FACE_FEATURE_SCHEMA_VERSION,
    FACE_NODE_FEATURE_KEYS,
)

GRAPH_JSON_KEYS = (
    "job_id",
    "building_id",
    "face_ids",
    "node_features",
    "edge_index",
    "edge_attr",
    "labels",
    "weights",
)


def hyperparams_from_settings(settings: Settings) -> dict[str, float | int]:
    return {
        "face_train_min_labeled_jobs": settings.face_train_min_labeled_jobs,
        "face_train_min_classes": settings.face_train_min_classes,
        "face_train_ratio": settings.face_train_ratio,
        "face_val_ratio": settings.face_val_ratio,
        "face_train_seed": settings.face_train_seed,
        "face_gnn_hidden_dim": settings.face_gnn_hidden_dim,
        "face_gnn_layers": settings.face_gnn_layers,
        "face_gnn_dropout": settings.face_gnn_dropout,
        "face_gnn_epochs": settings.face_gnn_epochs,
        "face_gnn_lr": settings.face_gnn_lr,
        "cavity_area_m2": settings.cavity_area_m2,
        "shaft_area_m2": settings.shaft_area_m2,
        "outdoor_area_m2": settings.outdoor_area_m2,
        "face_threshold": settings.face_threshold,
    }


def graph_to_json(graph: FaceGraphSample | dict[str, Any]) -> dict[str, Any]:
    payload = graph_to_dict(graph)
    return {key: payload[key] for key in GRAPH_JSON_KEYS}


def build_face_dataset_payload(
    *,
    rows: Sequence[TrainingRow],
    graphs: Sequence[FaceGraphSample | dict[str, Any]],
    settings: Settings,
    engine_sha: str,
) -> dict[str, Any]:
    eligibility = validate_face_dataset(
        rows,
        min_labeled_jobs=settings.face_train_min_labeled_jobs,
        min_classes=settings.face_train_min_classes,
    )
    return {
        "feature_schema_version": FACE_FEATURE_SCHEMA_VERSION,
        "edge_feature_schema_version": FACE_EDGE_FEATURE_SCHEMA_VERSION,
        "node_feature_keys": list(FACE_NODE_FEATURE_KEYS),
        "edge_feature_keys": list(FACE_EDGE_FEATURE_KEYS),
        "classes": list(FACE_CLASSES),
        "engine_sha": engine_sha,
        "hyperparams": hyperparams_from_settings(settings),
        "eligibility": {
            "ok": eligibility.ok,
            "labeled_jobs": eligibility.labeled_jobs,
            "n_classes": eligibility.n_classes,
            "n_labeled_faces": eligibility.n_labeled_faces,
            "reasons": list(eligibility.reasons),
        },
        "graphs": [graph_to_json(graph) for graph in graphs],
    }
