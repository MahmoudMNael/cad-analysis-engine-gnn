"""Immutable analysis snapshot writer (Postgres + object storage)."""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from shapely import wkt as shapely_wkt

from packages.common.errors import StorageError
from packages.contracts.dto.common import PointDto
from packages.contracts.dto.layout import FaceDto, LayoutDto
from packages.store.db.repositories.jobs import JobRepository
from packages.store.object_store import ObjectStore
from packages.views.reducer import reduce_view

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

    def load_json(self, job_id: str, filename: str) -> dict[str, Any]:
        key = self._objects.key_for_job(job_id, filename)
        return json.loads(self._objects.get_bytes(key))

    def load_document(self, job_id: str) -> dict[str, Any]:
        return self.load_json(job_id, "document.json")

    def load_plan_graph(self, job_id: str) -> dict[str, Any]:
        return self.load_json(job_id, "plan_graph.json")

    def load_derived_layout(self, job_id: str, events: Sequence[Any]) -> LayoutDto:
        physical = self.load_layout(job_id)
        return reduce_view(physical, events)

    def load_faces(self, job_id: str) -> list[FaceDto]:
        key = self._objects.key_for_job(job_id, "faces.parquet")
        if not self._objects.exists(key):
            raise StorageError(f"faces.parquet missing for {job_id}")
        table = pq.read_table(io.BytesIO(self._objects.get_bytes(key)))
        columns = {name: table.column(name).to_pylist() for name in table.column_names}
        n = len(columns.get("face_id") or [])
        faces: list[FaceDto] = []
        for i in range(n):
            raw_features = columns.get("features", [None] * n)[i]
            features = _parse_json_map(raw_features)
            geom = shapely_wkt.loads(str(columns["wkt"][i]))
            faces.append(
                FaceDto(
                    id=str(columns["face_id"][i]),
                    boundary=_ring_from_geom(geom),
                    area_m2=float(columns.get("area", [0.0] * n)[i] or 0.0),
                    predicted_class=str(columns.get("predicted_class", [""] * n)[i] or "unknown"),
                    confidence=float(columns.get("confidence", [0.0] * n)[i] or 0.0),
                    decision_source=str(columns.get("decision_source", [""] * n)[i] or "heuristic"),
                    accepted_as_room=bool(columns.get("accepted_as_room", [False] * n)[i]),
                    adjacent_unbounded=bool(float(features.get("adjacent_unbounded") or 0.0) >= 0.5),
                )
            )
        return faces


    def load_faces_columns(self, job_id: str) -> dict[str, list[Any]]:
        key = self._objects.key_for_job(job_id, "faces.parquet")
        if not self._objects.exists(key):
            raise StorageError(f"faces.parquet missing for {job_id}")
        table = pq.read_table(io.BytesIO(self._objects.get_bytes(key)))
        return {name: table.column(name).to_pylist() for name in table.column_names}


def _parse_json_map(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value:
        parsed = json.loads(value)
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _ring_from_geom(geom: Any) -> list[PointDto]:
    if geom.geom_type == "Polygon":
        coords = list(geom.exterior.coords)
    else:
        coords = list(getattr(geom, "coords", []))
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [PointDto(x=float(x), y=float(y)) for x, y, *_ in coords]
