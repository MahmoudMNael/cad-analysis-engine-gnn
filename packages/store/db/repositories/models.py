from __future__ import annotations

from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Json

from packages.contracts.enums import ModelRegistryStatus


class ModelRegistryRepository:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def ensure_heuristic(self, model_name: str, version: str, feature_schema_version: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_registry (
                    model_name, version, status, feature_schema_version, checkpoint_uri
                )
                VALUES (%s, %s, 'promoted', %s, %s)
                ON CONFLICT (model_name, version) DO NOTHING
                """,
                (model_name, version, feature_schema_version, f"heuristic:{version}"),
            )
        self._conn.commit()

    def promoted_version(self, model_name: str) -> str | None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT version FROM model_registry
                WHERE model_name = %s AND status = 'promoted'
                LIMIT 1
                """,
                (model_name,),
            )
            row = cur.fetchone()
        return row[0] if row else None

    def list_all(self) -> list[dict[str, Any]]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM model_registry ORDER BY model_name, created_at")
            return list(cur.fetchall())

    def get(self, model_name: str, version: str) -> dict[str, Any] | None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT * FROM model_registry
                WHERE model_name = %s AND version = %s
                """,
                (model_name, version),
            )
            return cur.fetchone()

    def get_promoted(self, model_name: str) -> dict[str, Any] | None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT * FROM model_registry
                WHERE model_name = %s AND status = 'promoted'
                LIMIT 1
                """,
                (model_name,),
            )
            return cur.fetchone()

    def checkpoint_uri(self, model_name: str, version: str) -> str | None:
        row = self.get(model_name, version)
        if not row:
            return None
        uri = row.get("checkpoint_uri")
        if not uri or str(uri).startswith("heuristic:"):
            return None
        return str(uri)

    def insert_candidate(self, **kwargs: Any) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_registry (
                    model_name, version, checkpoint_uri, metrics_json, status,
                    trained_on_engine_sha, feature_schema_version
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    kwargs["model_name"],
                    kwargs["version"],
                    kwargs.get("checkpoint_uri"),
                    Json(kwargs.get("metrics_json") or {}),
                    kwargs.get("status", ModelRegistryStatus.CANDIDATE.value),
                    kwargs.get("trained_on_engine_sha"),
                    kwargs["feature_schema_version"],
                ),
            )
        self._conn.commit()

    def promote(self, model_name: str, version: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE model_registry
                SET status = 'candidate'
                WHERE model_name = %s AND status = 'promoted' AND version <> %s
                """,
                (model_name, version),
            )
            cur.execute(
                """
                UPDATE model_registry
                SET status = 'promoted', promoted_at = NOW()
                WHERE model_name = %s AND version = %s
                """,
                (model_name, version),
            )
            if cur.rowcount != 1:
                self._conn.rollback()
                raise ValueError(f"unknown model {model_name} {version}")
        self._conn.commit()

    def reject(self, model_name: str, version: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE model_registry
                SET status = 'rejected'
                WHERE model_name = %s AND version = %s
                """,
                (model_name, version),
            )
            if cur.rowcount != 1:
                self._conn.rollback()
                raise ValueError(f"unknown model {model_name} {version}")
        self._conn.commit()

    def insert_training_run(
        self,
        run_id: str,
        model_name: str,
        *,
        status: str = "running",
        dataset_uri: str | None = None,
    ) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO training_runs (id, model_name, status, started_at, dataset_uri)
                VALUES (%s, %s, %s, NOW(), %s)
                """,
                (run_id, model_name, status, dataset_uri),
            )
        self._conn.commit()

    def finish_training_run(
        self,
        run_id: str,
        *,
        status: str,
        metrics_json: dict[str, Any] | None = None,
        error_detail: str | None = None,
    ) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE training_runs
                SET status = %s, completed_at = NOW(), metrics_json = %s, error_detail = %s
                WHERE id = %s
                """,
                (status, Json(metrics_json or {}), error_detail, run_id),
            )
        self._conn.commit()
