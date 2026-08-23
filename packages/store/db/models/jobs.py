from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class JobRecord:
    id: str
    source_hash: str
    status: str
    source_uri: str
    unit: str | None
    unit_source: str | None
    consent_training: bool
    engine_sha: str | None
    pipeline_version: str | None
    model_versions_json: dict[str, Any]
    configuration_hash: str | None
    error_code: str | None
    error_detail: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    attempt: int
    worker_id: str | None
    original_filename: str | None
    snapshot_version: int | None
    layout_rev: int = 0
