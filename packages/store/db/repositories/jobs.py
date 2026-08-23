from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Json

from packages.store.db.models.job import JobRecord


def _row(data: dict[str, Any]) -> JobRecord:
    return JobRecord(
        id=data["id"],
        source_hash=data["source_hash"],
        status=data["status"],
        source_uri=data["source_uri"],
        unit=data["unit"],
        unit_source=data["unit_source"],
        consent_training=data["consent_training"],
        engine_sha=data["engine_sha"],
        pipeline_version=data["pipeline_version"],
        model_versions_json=data["model_versions_json"] or {},
        configuration_hash=data["configuration_hash"],
        error_code=data["error_code"],
        error_detail=data["error_detail"],
        created_at=data["created_at"],
        started_at=data["started_at"],
        completed_at=data["completed_at"],
        attempt=data["attempt"],
        worker_id=data["worker_id"],
        original_filename=data["original_filename"],
        snapshot_version=data["snapshot_version"],
        layout_rev=int(data.get("layout_rev") or 0),
    )


class JobRepository:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def insert_queued(
        self,
        job_id: str,
        source_hash: str,
        source_uri: str,
        consent_training: bool,
        original_filename: str | None,
        configuration_hash: str,
        pipeline_version: str,
        engine_sha: str,
        model_versions: dict[str, str],
        unit: str | None = None,
        unit_source: str | None = None,
    ) -> JobRecord:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO jobs (
                    id, source_hash, status, source_uri, consent_training,
                    original_filename, configuration_hash, pipeline_version,
                    engine_sha, model_versions_json, unit, unit_source
                )
                VALUES (%s, %s, 'queued', %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    job_id,
                    source_hash,
                    source_uri,
                    consent_training,
                    original_filename,
                    configuration_hash,
                    pipeline_version,
                    engine_sha,
                    Json(model_versions),
                    unit,
                    unit_source,
                ),
            )
            row = cur.fetchone()
        self._conn.commit()
        assert row is not None
        return _row(row)

    def get(self, job_id: str) -> JobRecord | None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
            row = cur.fetchone()
        return _row(row) if row else None

    def list_queued(self) -> list[JobRecord]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at")
            rows = cur.fetchall()
        return [_row(r) for r in rows]

    def claim_queued(self, job_id: str, worker_id: str) -> JobRecord | None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                UPDATE jobs
                SET status = 'running',
                    started_at = NOW(),
                    worker_id = %s,
                    attempt = attempt + 1
                WHERE id = %s AND status = 'queued'
                RETURNING *
                """,
                (worker_id, job_id),
            )
            row = cur.fetchone()
        self._conn.commit()
        return _row(row) if row else None

    def mark_succeeded(
        self,
        job_id: str,
        unit: str,
        unit_source: str,
        snapshot_version: int,
    ) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE jobs
                SET status = 'succeeded',
                    completed_at = NOW(),
                    unit = %s,
                    unit_source = %s,
                    snapshot_version = %s,
                    error_code = NULL,
                    error_detail = NULL
                WHERE id = %s AND status = 'running'
                """,
                (unit, unit_source, snapshot_version, job_id),
            )
        self._conn.commit()

    def mark_failed(self, job_id: str, error_code: str, error_detail: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    completed_at = NOW(),
                    error_code = %s,
                    error_detail = %s
                WHERE id = %s AND status IN ('queued', 'running')
                """,
                (error_code, error_detail, job_id),
            )
        self._conn.commit()

    def reset_stale_running(self, timeout_seconds: int) -> list[str]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE jobs
                SET status = 'queued', worker_id = NULL
                WHERE status = 'running'
                  AND started_at < NOW() - (%s * INTERVAL '1 second')
                RETURNING id
                """,
                (timeout_seconds,),
            )
            rows = cur.fetchall()
        self._conn.commit()
        return [r[0] for r in rows]

    def add_artifact(
        self,
        job_id: str,
        snapshot_version: int,
        kind: str,
        uri: str,
        sha256: str,
    ) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO job_artifacts (job_id, snapshot_version, kind, uri, sha256)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (job_id, snapshot_version, kind)
                DO UPDATE SET uri = EXCLUDED.uri, sha256 = EXCLUDED.sha256
                """,
                (job_id, snapshot_version, kind, uri, sha256),
            )
        self._conn.commit()

    def list_artifacts(self, job_id: str, snapshot_version: int = 1) -> dict[str, dict[str, str]]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT kind, uri, sha256
                FROM job_artifacts
                WHERE job_id = %s AND snapshot_version = %s
                """,
                (job_id, snapshot_version),
            )
            rows = cur.fetchall()
        return {r["kind"]: {"uri": r["uri"], "sha256": r["sha256"]} for r in rows}

    def layout_rev(self, job_id: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT layout_rev FROM jobs WHERE id = %s", (job_id,))
            row = cur.fetchone()
        if row is not None and row[0] is not None:
            return int(row[0])
        return 0

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def list_succeeded(self) -> list[JobRecord]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM jobs WHERE status = 'succeeded' ORDER BY created_at")
            rows = cur.fetchall()
        return [_row(r) for r in rows]
