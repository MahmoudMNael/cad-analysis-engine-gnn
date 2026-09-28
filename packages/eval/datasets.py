"""Training-row schema and job/building splits. Never split by face."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from shapely import wkt as shapely_wkt
from shapely.errors import GEOSException

from packages.common.hashing import stable_hash
from packages.contracts.enums import FaceClass, FeedbackAction, LabelSource
from packages.models_nn.common import FACE_CLASSES, FACE_CLASS_TO_INDEX
from packages.models_nn.features.face import build_face_adjacency, vectorize_face_features

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
    if len(buildings) > 1 and not val_buildings:
        val_buildings.add(buildings[-1])
        train_buildings.discard(buildings[-1])
    train = [row for row in rows if (row.building_id or row.job_id) in train_buildings]
    val = [row for row in rows if (row.building_id or row.job_id) in val_buildings]
    return train, val


FACE_LABEL_ALIASES: dict[str, str] = {
    "not_indoor": FaceClass.CAVITY.value,
    "not-indoor": FaceClass.CAVITY.value,
    "false_room": FaceClass.CAVITY.value,
}

EXPLICIT_FACE_SOURCES = {LabelSource.EXPERT, LabelSource.USER_EXPLICIT}


@dataclass
class FaceGraphSample:
    """One building/job face graph. Labels are per-node; unlabeled nodes are -1."""

    job_id: str
    building_id: str
    face_ids: list[str]
    node_features: list[list[float]]
    edge_index: list[tuple[int, int]]
    edge_attr: list[list[float]]
    labels: list[int]
    weights: list[float]


@dataclass(frozen=True)
class DatasetEligibility:
    ok: bool
    labeled_jobs: int
    n_classes: int
    n_labeled_faces: int
    reasons: tuple[str, ...]


def normalize_face_label(label: str) -> str | None:
    key = str(label).strip().lower()
    if key in FACE_CLASSES:
        return key
    return FACE_LABEL_ALIASES.get(key)


def collapse_face_labels(rows: Sequence[TrainingRow]) -> dict[tuple[str, str], TrainingRow]:
    """Latest explicit label per (job_id, target_id). Weak rows never overwrite."""
    chosen: dict[tuple[str, str], TrainingRow] = {}
    for row in rows:
        if row.weight <= 0 or row.label_source == LabelSource.WEAK:
            continue
        mapped = normalize_face_label(row.label)
        if mapped is None:
            continue
        key = (row.job_id, row.target_id)
        chosen[key] = replace(row, label=mapped)
    return chosen


def explicit_face_job_ids(rows: Sequence[TrainingRow]) -> set[str]:
    return {
        row.job_id
        for row in collapse_face_labels(rows).values()
        if row.label_source in EXPLICIT_FACE_SOURCES
    }


def validate_face_dataset(
    rows: Sequence[TrainingRow],
    *,
    min_labeled_jobs: int = 50,
    min_classes: int = 2,
) -> DatasetEligibility:
    collapsed = collapse_face_labels(rows)
    jobs = {row.job_id for row in collapsed.values() if row.label_source in EXPLICIT_FACE_SOURCES}
    classes = {row.label for row in collapsed.values()}
    reasons: list[str] = []
    if len(jobs) < min_labeled_jobs:
        reasons.append(
            f"need at least {min_labeled_jobs} explicitly labeled CAD files, have {len(jobs)}"
        )
    if len(classes) < min_classes:
        reasons.append(f"need at least {min_classes} face classes, have {len(classes)}")
    if not collapsed:
        reasons.append("no explicit face labels after filtering weak/unmapped rows")
    return DatasetEligibility(
        ok=not reasons,
        labeled_jobs=len(jobs),
        n_classes=len(classes),
        n_labeled_faces=len(collapsed),
        reasons=tuple(reasons),
    )


def _bucket(seed: int, building: str) -> float:
    digest = hashlib.sha256(f"{seed}:{building}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def split_by_building_held_out(
    samples: Sequence[FaceGraphSample],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 0,
) -> tuple[list[FaceGraphSample], list[FaceGraphSample], list[FaceGraphSample]]:
    """Split graphs by building, never by face. Returns train, val, held-out test."""
    buildings = sorted({sample.building_id or sample.job_id for sample in samples})
    train_b: set[str] = set()
    val_b: set[str] = set()
    test_b: set[str] = set()
    val_cut = train_ratio
    test_cut = train_ratio + val_ratio
    for building in buildings:
        bucket = _bucket(seed, building)
        if bucket < val_cut:
            train_b.add(building)
        elif bucket < test_cut:
            val_b.add(building)
        else:
            test_b.add(building)
    if len(buildings) >= 3:
        if not train_b:
            train_b.add(buildings[0])
            val_b.discard(buildings[0])
            test_b.discard(buildings[0])
        if not val_b:
            donor = next((b for b in buildings if b in train_b and len(train_b) > 1), buildings[-2])
            val_b.add(donor)
            train_b.discard(donor)
            test_b.discard(donor)
        if not test_b:
            donor = next((b for b in buildings if b in train_b and len(train_b) > 1), buildings[-1])
            test_b.add(donor)
            train_b.discard(donor)
            val_b.discard(donor)
    elif len(buildings) == 2:
        train_b = {buildings[0]}
        val_b = {buildings[1]}
        test_b = {buildings[1]}
    elif len(buildings) == 1:
        train_b = {buildings[0]}
        val_b = set()
        test_b = set()

    def _select(bucket: set[str]) -> list[FaceGraphSample]:
        return [sample for sample in samples if (sample.building_id or sample.job_id) in bucket]

    return _select(train_b), _select(val_b), _select(test_b)


def face_graphs_from_rows(
    rows: Sequence[TrainingRow],
    *,
    faces_columns_by_job: Mapping[str, Mapping[str, list[Any]]] | None = None,
) -> list[FaceGraphSample]:
    """Build per-job face graphs. Snapshot columns supply unlabeled neighbors and WKT."""
    collapsed = collapse_face_labels(rows)
    jobs: dict[str, list[TrainingRow]] = {}
    for row in collapsed.values():
        jobs.setdefault(row.job_id, []).append(row)
    columns_by_job = faces_columns_by_job or {}
    graphs: list[FaceGraphSample] = []
    for job_id, labeled in jobs.items():
        building_id = labeled[0].building_id or job_id
        columns = columns_by_job.get(job_id)
        graphs.append(_graph_for_job(job_id, building_id, labeled, columns))
    return graphs


def _graph_for_job(
    job_id: str,
    building_id: str,
    labeled: Sequence[TrainingRow],
    columns: Mapping[str, list[Any]] | None,
) -> FaceGraphSample:
    label_by_face = {row.target_id: row for row in labeled}
    face_ids: list[str] = []
    features: list[dict[str, Any]] = []
    geoms: list[Any] = []
    areas: list[float] = []
    perims: list[float] = []
    if columns:
        ids = [str(v) for v in (columns.get("face_id") or [])]
        raw_features = columns.get("features") or [None] * len(ids)
        raw_wkt = columns.get("wkt") or [None] * len(ids)
        raw_area = columns.get("area") or [None] * len(ids)
        raw_perim = columns.get("perimeter") or [None] * len(ids)
        for i, face_id in enumerate(ids):
            face_ids.append(face_id)
            feats = _parse_features(raw_features[i] if i < len(raw_features) else None)
            features.append(feats)
            geom = _parse_wkt(raw_wkt[i] if i < len(raw_wkt) else None)
            geoms.append(geom)
            areas.append(
                float(raw_area[i] if i < len(raw_area) and raw_area[i] is not None else feats.get("area") or 0.0)
            )
            perims.append(
                float(
                    raw_perim[i]
                    if i < len(raw_perim) and raw_perim[i] is not None
                    else feats.get("perimeter") or 0.0
                )
            )
    missing = [face_id for face_id in label_by_face if face_id not in set(face_ids)]
    for face_id in missing:
        row = label_by_face[face_id]
        face_ids.append(face_id)
        features.append(dict(row.features))
        geoms.append(None)
        areas.append(float(row.features.get("area") or 0.0))
        perims.append(float(row.features.get("perimeter") or 0.0))
    if not face_ids:
        return FaceGraphSample(
            job_id=job_id,
            building_id=building_id,
            face_ids=[],
            node_features=[],
            edge_index=[],
            edge_attr=[],
            labels=[],
            weights=[],
        )
    node_features = [vectorize_face_features(feats) for feats in features]
    labels: list[int] = []
    weights: list[float] = []
    for face_id in face_ids:
        row = label_by_face.get(face_id)
        if row is None:
            labels.append(-1)
            weights.append(0.0)
            continue
        labels.append(FACE_CLASS_TO_INDEX[row.label])
        weights.append(float(row.weight))
    edge_index: list[tuple[int, int]] = []
    edge_attr: list[list[float]] = []
    if any(geom is not None for geom in geoms):
        payload = []
        for face_id, geom, area, perim in zip(face_ids, geoms, areas, perims, strict=False):
            if geom is None:
                continue
            payload.append((face_id, geom, area, perim))
        id_to_index = {face_id: i for i, face_id in enumerate(face_ids)}
        raw_index, raw_attr = build_face_adjacency(payload)
        for (src, dst), attr in zip(raw_index, raw_attr, strict=False):
            src_id = payload[src][0]
            dst_id = payload[dst][0]
            edge_index.append((id_to_index[src_id], id_to_index[dst_id]))
            edge_attr.append(attr)
    return FaceGraphSample(
        job_id=job_id,
        building_id=building_id,
        face_ids=face_ids,
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        labels=labels,
        weights=weights,
    )


def training_row_from_record(record: Mapping[str, Any]) -> TrainingRow:
    features = record.get("features") or {}
    if isinstance(features, str):
        features = json.loads(features) if features else {}
    source = record["label_source"]
    if not isinstance(source, LabelSource):
        source = LabelSource(str(source))
    return TrainingRow(
        sample_id=str(record["sample_id"]),
        job_id=str(record["job_id"]),
        building_id=str(record["building_id"]),
        target_id=str(record["target_id"]),
        feature_schema_version=str(record["feature_schema_version"]),
        features=dict(features),
        label=str(record["label"]),
        label_source=source,
        weight=float(record["weight"]),
        engine_sha=str(record["engine_sha"]),
        model_input_version=str(record["model_input_version"]),
    )


def read_training_rows(path: Path) -> list[TrainingRow]:
    table = pq.read_table(path)
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    n = len(columns.get("sample_id") or [])
    return [training_row_from_record({field: columns[field][i] for field in TRAINING_ROW_FIELDS}) for i in range(n)]


def read_compiled_lake(directory: Path) -> CompiledLake:
    return CompiledLake(
        face_cls=read_training_rows(directory / "face_cls.parquet")
        if (directory / "face_cls.parquet").is_file()
        else [],
        edge_cls=read_training_rows(directory / "edge_cls.parquet")
        if (directory / "edge_cls.parquet").is_file()
        else [],
        link_pred=read_training_rows(directory / "link_pred.parquet")
        if (directory / "link_pred.parquet").is_file()
        else [],
    )


def _parse_features(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value:
        parsed = json.loads(value)
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _parse_wkt(value: Any) -> Any | None:
    if not value:
        return None
    try:
        return shapely_wkt.loads(str(value))
    except (TypeError, ValueError, GEOSException):
        return None


def _write_rows(path: Path, rows: list[TrainingRow]) -> None:
    payload: dict[str, list[Any]] = {field: [] for field in TRAINING_ROW_FIELDS}
    for row in rows:
        record = row.to_record()
        for field in TRAINING_ROW_FIELDS:
            payload[field].append(record[field])
    table = pa.table(payload)
    pq.write_table(table, path)
