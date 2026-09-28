"""Evaluation hooks for GNN-C and later phases."""

from packages.eval.datasets import (
    CompiledLake,
    FaceGraphSample,
    TrainingRow,
    split_by_building,
    split_by_building_held_out,
    validate_face_dataset,
)
from packages.eval.metrics import face_classification_metrics, flatten_metrics
from packages.eval.promotion import promotion_gate

__all__ = [
    "CompiledLake",
    "FaceGraphSample",
    "TrainingRow",
    "face_classification_metrics",
    "flatten_metrics",
    "promotion_gate",
    "split_by_building",
    "split_by_building_held_out",
    "validate_face_dataset",
]
