"""Idempotent job/snapshot guards."""

from __future__ import annotations

from packages.store.db.models.job import JobRecord
from packages.store.object_store import ObjectStore

REQUIRED_KINDS = ("source", "document", "plan_graph", "faces", "layout", "meta")


def snapshot_is_complete(job: JobRecord, store: ObjectStore, artifacts: dict[str, dict[str, str]]) -> bool:
    if job.status == "succeeded" and store.snapshot_complete(job.id):
        return True
    if not store.snapshot_complete(job.id):
        return False
    return all(kind in artifacts for kind in REQUIRED_KINDS)
