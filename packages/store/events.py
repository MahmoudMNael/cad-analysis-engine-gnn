"""Append-only feedback events. Persistence is used from P2; schema exists in P1."""

from __future__ import annotations

from typing import Any


class EventStore:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def list_for_job(self, job_id: str) -> list[dict[str, Any]]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, job_id, sequence, ts, actor, action, target_ids,
                       payload_json, layout_rev, retracted_by_event_id
                FROM feedback_events
                WHERE job_id = %s
                ORDER BY sequence
                """,
                (job_id,),
            )
            columns = [
                "id",
                "job_id",
                "sequence",
                "ts",
                "actor",
                "action",
                "target_ids",
                "payload_json",
                "layout_rev",
                "retracted_by_event_id",
            ]
            return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]
