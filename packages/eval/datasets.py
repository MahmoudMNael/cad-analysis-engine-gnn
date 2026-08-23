"""Training-row schema and job/building splits. Never split by face."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from packages.contracts.enums import FeedbackAction, LabelSource
from packages.common.hashing import stable_hash

TRAINING_ROW_FIELDS = (
    "sample_id",
    "job_id",
    "building_id",
    "target_id",
    "feature_schema_version",
    "features",
    "label",
    "label_source",
    "weight",
    "engine_sha",
    "model_input_version",
)

WEIGHT_EXPERT = 1.0
WEIGHT_USER_EXPLICIT = 0.75
WEIGHT_SYNTHETIC = 0.5
WEIGHT_WEAK = 0.0

DEFAULT_WEIGHTS: dict[LabelSource, float] = {
    LabelSource.EXPERT: WEIGHT_EXPERT,
    LabelSource.USER_EXPLICIT: WEIGHT_USER_EXPLICIT,
    LabelSource.SYNTHETIC: WEIGHT_SYNTHETIC,
    LabelSource.WEAK: WEIGHT_WEAK,
}


@dataclass(frozen=True)
class TrainingRow:
    sample_id: str
    job_id: str
    building_id: str
    target_id: str
    feature_schema_version: str
    features: dict[str, Any]
    label: str
    label_source: LabelSource
    weight: float
    engine_sha: str
    model_input_version: str

    def to_record(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "job_id": self.job_id,
            "building_id": self.building_id,
            "target_id": self.target_id,
            "feature_schema_version": self.feature_schema_version,
            "features": json.dumps(self.features, sort_keys=True, default=str),
            "label": self.label,
            "label_source": self.label_source.value,
            "weight": float(self.weight),
            "engine_sha": self.engine_sha,
            "model_input_version": self.model_input_version,
        }


@dataclass
class CompiledLake:
    face_cls: list[TrainingRow]
    edge_cls: list[TrainingRow]
    link_pred: list[TrainingRow]

    def write_parquet(self, directory: Path) -> dict[str, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "face_cls": directory / "face_cls.parquet",
            "edge_cls": directory / "edge_cls.parquet",
            "link_pred": directory / "link_pred.parquet",
        }
        _write_rows(paths["face_cls"], self.face_cls)
        _write_rows(paths["edge_cls"], self.edge_cls)
        _write_rows(paths["link_pred"], self.link_pred)
        return paths


def label_source_for(actor: str, action: FeedbackAction | str) -> LabelSource:
    action_value = action.value if isinstance(action, FeedbackAction) else str(action)
    if action_value == FeedbackAction.ACCEPT.value:
        return LabelSource.WEAK
    actor_key = actor.strip().lower()
    if actor_key == "expert":
        return LabelSource.EXPERT
    if actor_key == "synthetic":
        return LabelSource.SYNTHETIC
    if actor_key == "weak":
        return LabelSource.WEAK
    return LabelSource.USER_EXPLICIT


def weight_for(source: LabelSource, weights: dict[LabelSource, float] | None = None) -> float:
    table = weights or DEFAULT_WEIGHTS
    return float(table[source])


def make_sample_id(
    *,
    job_id: str,
    dataset: str,
    target_id: str,
    sequence: int,
    action: str,
) -> str:
    return stable_hash(
        "training_row",
        [],
        {
            "job_id": job_id,
            "dataset": dataset,
            "target_id": target_id,
            "sequence": sequence,
            "action": action,
        },
    )


def split_by_building(
    rows: Sequence[TrainingRow],
    *,
    train_ratio: float = 0.8,
    seed: int = 0,
) -> tuple[list[TrainingRow], list[TrainingRow]]:
    """Split by job/building, never by face."""
    buildings = sorted({row.building_id or row.job_id for row in rows})
    train_buildings: set[str] = set()
    val_buildings: set[str] = set()
    for building in buildings:
        digest = hashlib.sha256(f"{seed}:{building}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") / float(2**64)
        if bucket < train_ratio:
            train_buildings.add(building)
        else:
            val_buildings.add(building)
    if buildings and not train_buildings:
        train_buildings.add(buildings[0])
        val_buildings.discard(buildings[0])
    train = [row for row in rows if (row.building_id or row.job_id) in train_buildings]
    val = [row for row in rows if (row.building_id or row.job_id) in val_buildings]
    return train, val


def _write_rows(path: Path, rows: list[TrainingRow]) -> None:
    payload: dict[str, list[Any]] = {field: [] for field in TRAINING_ROW_FIELDS}
    for row in rows:
        record = row.to_record()
        for field in TRAINING_ROW_FIELDS:
            payload[field].append(record[field])
    table = pa.table(payload)
    pq.write_table(table, path)
