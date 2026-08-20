"""Immutable analysis snapshot writer (Postgres + object storage)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from packages.common.errors import StorageError
from packages.contracts.dto.layout import LayoutDto
from packages.store.db.repositories.jobs import JobRepository
from packages.store.object_store import ObjectStore

SNAPSHOT_FILES = {
    "source": "source.dxf",
    "document": "document.json",
    "plan_graph": "plan_graph.json",
    "faces": "faces.parquet",
    "layout": "layout.json",
    "meta": "meta.json",
}


def _dumps(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, default=str).encode("utf-8")


class SnapshotStore:
    def __init__(self, object_store: ObjectStore, jobs: JobRepository) -> None:
        self._objects = object_store
        self._jobs = jobs

    def write(
        self,
        job_id: str,
        source_bytes: bytes,
        document_payload: dict[str, Any],
        plan_graph_payload: dict[str, Any],
        faces_rows: dict[str, list[Any]],
        layout: LayoutDto,
        meta: dict[str, Any],
        snapshot_version: int = 1,
    ) -> dict[str, tuple[str, str]]:
        layout_json = layout.model_dump(mode="json")
        LayoutDto.model_validate(layout_json)

        written: dict[str, tuple[str, str]] = {}
        blobs: dict[str, bytes] = {
            "source": source_bytes,
            "document": _dumps(document_payload),
            "plan_graph": _dumps(plan_graph_payload),
            "layout": _dumps(layout_json),
            "meta": _dumps(meta),
        }
        for kind, data in blobs.items():
            key = self._objects.key_for_job(job_id, SNAPSHOT_FILES[kind])
            uri, digest = self._objects.put_bytes(key, data)
            written[kind] = (uri, digest)
            self._jobs.add_artifact(job_id, snapshot_version, kind, uri, digest)

        table = pa.table(faces_rows)
        faces_key = self._objects.key_for_job(job_id, SNAPSHOT_FILES["faces"])
        path = self._objects._resolve(faces_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        digest = self._objects.digest_file(faces_key)
        uri = self._objects.uri_for(faces_key)
        written["faces"] = (uri, digest)
        self._jobs.add_artifact(job_id, snapshot_version, "faces", uri, digest)

        if not self._objects.snapshot_complete(job_id):
            raise StorageError("snapshot artifacts incomplete")
        return written

    def load_layout(self, job_id: str) -> LayoutDto:
        key = self._objects.key_for_job(job_id, "layout.json")
        if not self._objects.exists(key):
            raise StorageError(f"layout.json missing for {job_id}")
        payload = json.loads(self._objects.get_bytes(key))
        return LayoutDto.model_validate(payload)

    def load_meta(self, job_id: str) -> dict[str, Any]:
        key = self._objects.key_for_job(job_id, "meta.json")
        return json.loads(self._objects.get_bytes(key))

    def load_faces_present(self, job_id: str) -> bool:
        return self._objects.exists(self._objects.key_for_job(job_id, "faces.parquet"))
