"""Compile consented face labels into a notebook-ready JSON graph bundle."""

from __future__ import annotations

from typing import Any

from packages.common.configuration import Settings, get_settings
from packages.eval.datasets import face_graphs_from_rows
from packages.eval.face_bundle import build_face_dataset_payload
from packages.store.object_store import ObjectStore


def export_face_dataset(
    *,
    conn: Any,
    object_store: ObjectStore,
    settings: Settings | None = None,
    engine_sha: str = "",
) -> dict[str, Any]:
    from apps.trainer.compile import compile_lake
    from apps.trainer.train import load_faces_columns

    cfg = settings or get_settings()
    lake = compile_lake(conn=conn, object_store=object_store)
    job_ids = {row.job_id for row in lake.face_cls}
    columns = load_faces_columns(conn, object_store, job_ids)
    graphs = face_graphs_from_rows(lake.face_cls, faces_columns_by_job=columns)
    sha = engine_sha or cfg.engine_sha_override or ""
    return build_face_dataset_payload(
        rows=lake.face_cls,
        graphs=graphs,
        settings=cfg,
        engine_sha=sha,
    )
