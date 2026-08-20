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
