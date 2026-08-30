"""GNN-C training loop. Registers a candidate; never promotes the checkpoint."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from packages.common.configuration import Settings, get_settings
from packages.common.errors import InvalidRequestError
from packages.common.hashing import engine_sha_from_sources
from packages.common.logging import get_logger
from packages.eval.datasets import (
    FaceGraphSample,
    face_graphs_from_rows,
    split_by_building_held_out,
    validate_face_dataset,
)
from packages.eval.face_bundle import hyperparams_from_settings
from packages.eval.metrics import flatten_metrics
from packages.models_nn.features.face import FACE_FEATURE_SCHEMA_VERSION
from packages.models_nn.registry import checkpoint_object_key
from packages.store.db.repositories.jobs import JobRepository
from packages.store.db.repositories.models import ModelRegistryRepository
from packages.store.object_store import ObjectStore
from packages.store.snapshots import SnapshotStore

log = get_logger("trainer.train")

NOTEBOOK_TRAIN_HINT = (
    "PyTorch is not installed. Train GNN-C in notebooks/gnn_c_train.ipynb "
    "(fetch kernel + dataset from the local admin API after tunneling)."
)


@dataclass
class TrainResult:
    version: str
    checkpoint_uri: str | None
    metrics: dict[str, Any]
    eligible: bool
    reasons: tuple[str, ...]
    training_run_id: str
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "checkpoint_uri": self.checkpoint_uri,
            "metrics": flatten_metrics(self.metrics),
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "training_run_id": self.training_run_id,
            "status": self.status,
        }


def new_face_version(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"gnn:face:{stamp}"


def train_face_on_graphs(
    train_graphs: Sequence[FaceGraphSample],
    val_graphs: Sequence[FaceGraphSample],
    *,
    settings: Settings,
    engine_sha: str,
) -> tuple[Any, dict[str, Any]]:
    from packages.models_nn.face_train_kernel import train_face_on_graphs as train_kernel
    from packages.models_nn.face_train_kernel import torch_available

    if not torch_available():
        raise ImportError(NOTEBOOK_TRAIN_HINT)
    return train_kernel(
        train_graphs,
        val_graphs,
        hyperparams=hyperparams_from_settings(settings),
        engine_sha=engine_sha,
    )


def train_face(
    *,
    conn: Any,
    object_store: ObjectStore,
    settings: Settings | None = None,
    engine_sha: str | None = None,
    rows: Sequence[Any] | None = None,
    graphs: Sequence[FaceGraphSample] | None = None,
    lake_dir: Any | None = None,
) -> TrainResult:
    cfg = settings or get_settings()
    run_id = str(uuid.uuid4())
    registry = ModelRegistryRepository(conn)
    registry.insert_training_run(run_id, "face", status="running")
    try:
        if rows is None:
            from apps.trainer.compile import compile_lake

            lake = compile_lake(conn=conn, object_store=object_store, output_dir=lake_dir)
            rows = lake.face_cls
        eligibility = validate_face_dataset(
            rows,
            min_labeled_jobs=cfg.face_train_min_labeled_jobs,
            min_classes=cfg.face_train_min_classes,
        )
        if not eligibility.ok:
            registry.finish_training_run(
                run_id,
                status="failed",
                error_detail="; ".join(eligibility.reasons),
            )
            return TrainResult(
                version="",
                checkpoint_uri=None,
                metrics={},
                eligible=False,
                reasons=eligibility.reasons,
                training_run_id=run_id,
                status="failed",
            )
        if graphs is None:
            faces_columns = load_faces_columns(conn, object_store, {row.job_id for row in rows})
            graphs = face_graphs_from_rows(rows, faces_columns_by_job=faces_columns)
        train_g, val_g, test_g = split_by_building_held_out(
            graphs,
            train_ratio=cfg.face_train_ratio,
            val_ratio=cfg.face_val_ratio,
            seed=cfg.face_train_seed,
        )
        sha = engine_sha or cfg.engine_sha_override or _engine_sha()
        model, train_info = train_face_on_graphs(train_g, val_g, settings=cfg, engine_sha=sha)
        from apps.trainer.evaluate import evaluate_face_checkpoint

        metrics = evaluate_face_checkpoint(model, test_g or val_g, train_graphs=train_g, settings=cfg)
        metrics["best_val_macro_f1"] = train_info["best_val_macro_f1"]
        held_out_ok = bool((test_g or val_g) and metrics.get("n", 0) > 0)
        metrics["held_out_passed"] = 1.0 if held_out_ok else 0.0
        metrics["n_train_graphs"] = float(len(train_g))
        metrics["n_val_graphs"] = float(len(val_g))
        metrics["n_test_graphs"] = float(len(test_g))
        version = new_face_version()
        key = checkpoint_object_key("face", version)
        uri, _digest = object_store.put_bytes(key, model.save_bytes())
        registry.insert_candidate(
            model_name="face",
            version=version,
            checkpoint_uri=uri,
            metrics_json=metrics,
            trained_on_engine_sha=sha,
            feature_schema_version=FACE_FEATURE_SCHEMA_VERSION,
        )
        registry.finish_training_run(run_id, status="succeeded", metrics_json=flatten_metrics(metrics))
        return TrainResult(
            version=version,
            checkpoint_uri=uri,
            metrics=metrics,
            eligible=True,
            reasons=(),
            training_run_id=run_id,
            status="candidate",
        )
    except Exception as exc:
        registry.finish_training_run(run_id, status="failed", error_detail=str(exc)[:500])
        raise


def register_imported_face_checkpoint(
    *,
    conn: Any,
    object_store: ObjectStore,
    checkpoint_bytes: bytes,
    metrics: Mapping[str, Any],
    engine_sha: str | None = None,
) -> dict[str, Any]:
    """Store a notebook checkpoint as a registry candidate. Never promotes."""
    schema = str(metrics.get("feature_schema_version") or "")
    if schema != FACE_FEATURE_SCHEMA_VERSION:
        raise InvalidRequestError(
            f"feature_schema_version must be {FACE_FEATURE_SCHEMA_VERSION}",
            fields={"feature_schema_version": schema or "missing"},
        )
    if not checkpoint_bytes:
        raise InvalidRequestError("empty checkpoint")
    _validate_checkpoint_bytes(checkpoint_bytes)
    version = new_face_version()
    sha = engine_sha or str(metrics.get("engine_sha") or "") or None
    key = checkpoint_object_key("face", version)
    uri, _digest = object_store.put_bytes(key, checkpoint_bytes)
    registry = ModelRegistryRepository(conn)
    registry.insert_candidate(
        model_name="face",
        version=version,
        checkpoint_uri=uri,
        metrics_json=dict(metrics),
        trained_on_engine_sha=sha,
        feature_schema_version=FACE_FEATURE_SCHEMA_VERSION,
        status="candidate",
    )
    row = registry.get("face", version)
    if row is None:
        raise InvalidRequestError("candidate was not stored")
    return row


def load_faces_columns(conn: Any, object_store: ObjectStore, job_ids: set[str]) -> dict[str, dict[str, list[Any]]]:
    jobs = JobRepository(conn)
    snapshots = SnapshotStore(object_store, jobs)
    out: dict[str, dict[str, list[Any]]] = {}
    for job_id in job_ids:
        try:
            if snapshots.load_faces_present(job_id):
                out[job_id] = snapshots.load_faces_columns(job_id)
        except Exception:
            continue
    return out


def _validate_checkpoint_bytes(data: bytes) -> None:
    from packages.models_nn.face_train_kernel import FaceGNN, torch_available

    if not torch_available():
        return
    try:
        FaceGNN.load_bytes(data)
    except Exception as exc:
        raise InvalidRequestError(f"checkpoint could not be loaded: {exc}") from exc


def _engine_sha() -> str:
    from pathlib import Path

    return engine_sha_from_sources(Path(__file__).resolve().parents[2])


def main() -> None:
    from packages.models_nn.face_train_kernel import torch_available

    if not torch_available():
        print(NOTEBOOK_TRAIN_HINT)
        raise SystemExit(2)

    import psycopg

    settings = get_settings()
    store = ObjectStore(settings.object_store_dir)
    with psycopg.connect(settings.database_url) as conn:
        result = train_face(conn=conn, object_store=store, settings=settings)
    payload = result.as_dict()
    print(json.dumps(payload, indent=2))
    if not result.eligible:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
