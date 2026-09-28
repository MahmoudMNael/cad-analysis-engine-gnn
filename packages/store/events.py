"""Append-only feedback events with monotonic sequence and layout revisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Json

from packages.common.errors import JobNotFoundError, LayoutRevisionConflictError
from packages.common.hashing import stable_hash
from packages.contracts.dto.feedback import FeedbackEventDto
from packages.contracts.enums import FeedbackAction


@dataclass(frozen=True)
class EventAppend:
    action: str
    target_ids: list[str]
    payload: dict[str, Any]
    actor: str
    retract_event_id: str | None = None
    clear_retraction_id: str | None = None


def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value)


def _json_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return {}


def row_to_dto(row: dict[str, Any]) -> FeedbackEventDto:
    ts = row["ts"]
    if isinstance(ts, datetime):
        stamp = ts.isoformat()
    else:
        stamp = str(ts)
    return FeedbackEventDto(
        event_id=str(row["id"]),
        job_id=str(row["job_id"]),
        sequence=int(row["sequence"]),
        ts=stamp,
        actor=str(row["actor"]),
        action=FeedbackAction(str(row["action"])),
        target_ids=[str(x) for x in _json_list(row.get("target_ids"))],
        payload=_json_dict(row.get("payload_json")),
        layout_rev=int(row["layout_rev"]),
        retracted_by_event_id=(
            str(row["retracted_by_event_id"]) if row.get("retracted_by_event_id") else None
        ),
    )


class EventStore:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def list_for_job(self, job_id: str) -> list[dict[str, Any]]:
        with self._conn.cursor(row_factory=dict_row) as cur:
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
            return [dict(row) for row in cur.fetchall()]

    def list_dtos(self, job_id: str) -> list[FeedbackEventDto]:
        return [row_to_dto(row) for row in self.list_for_job(job_id)]

    def reset_job(self, job_id: str) -> None:
        """Drop all feedback for a job and return it to layout_rev 0."""
        with self._conn.cursor() as cur:
            cur.execute("SELECT id FROM jobs WHERE id = %s FOR UPDATE", (job_id,))
            locked = cur.fetchone()
            if locked is None:
                self._conn.rollback()
                raise JobNotFoundError(job_id)
            cur.execute("DELETE FROM layout_revisions WHERE job_id = %s", (job_id,))
            cur.execute("DELETE FROM feedback_events WHERE job_id = %s", (job_id,))
            cur.execute("UPDATE jobs SET layout_rev = 0 WHERE id = %s", (job_id,))
        self._conn.commit()

    def append_batch(
        self,
        job_id: str,
        records: list[EventAppend],
        expected_layout_rev: int | None = None,
    ) -> list[FeedbackEventDto]:
        if not records:
            return []
        written: list[FeedbackEventDto] = []
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT layout_rev FROM jobs WHERE id = %s FOR UPDATE", (job_id,))
            locked = cur.fetchone()
            if locked is None:
                self._conn.rollback()
                raise LookupError(job_id)
            current_rev = int(locked["layout_rev"] or 0)
            if expected_layout_rev is not None and expected_layout_rev != current_rev:
                self._conn.rollback()
                raise LayoutRevisionConflictError(job_id, expected_layout_rev, current_rev)
            cur.execute(
                "SELECT COALESCE(MAX(sequence), 0) AS seq FROM feedback_events WHERE job_id = %s",
                (job_id,),
            )
            sequence = int(cur.fetchone()["seq"])
            for record in records:
                sequence += 1
                current_rev += 1
                event_id = stable_hash(
                    "feedback_event",
                    [],
                    {
                        "job_id": job_id,
                        "sequence": sequence,
                        "action": record.action,
                        "target_ids": list(record.target_ids),
                        "payload": record.payload,
                        "actor": record.actor,
                    },
                )
                cur.execute(
                    """
                    INSERT INTO feedback_events (
                        id, job_id, sequence, actor, action, target_ids,
                        payload_json, layout_rev
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, job_id, sequence, ts, actor, action, target_ids,
                              payload_json, layout_rev, retracted_by_event_id
                    """,
                    (
                        event_id,
                        job_id,
                        sequence,
                        record.actor,
                        record.action,
                        Json(list(record.target_ids)),
                        Json(dict(record.payload)),
                        current_rev,
                    ),
                )
                row = cur.fetchone()
                assert row is not None
                if record.retract_event_id:
                    cur.execute(
                        """
                        UPDATE feedback_events
                        SET retracted_by_event_id = %s
                        WHERE id = %s AND job_id = %s AND retracted_by_event_id IS NULL
                        """,
                        (event_id, record.retract_event_id, job_id),
                    )
                if record.clear_retraction_id:
                    cur.execute(
                        """
                        UPDATE feedback_events
                        SET retracted_by_event_id = NULL
                        WHERE id = %s AND job_id = %s
                        """,
                        (record.clear_retraction_id, job_id),
                    )
                cur.execute(
                    """
                    INSERT INTO layout_revisions (job_id, layout_rev, event_id)
                    VALUES (%s, %s, %s)
                    """,
                    (job_id, current_rev, event_id),
                )
                written.append(row_to_dto(dict(row)))
            cur.execute(
                "UPDATE jobs SET layout_rev = %s WHERE id = %s",
                (current_rev, job_id),
            )
        self._conn.commit()
        return written
