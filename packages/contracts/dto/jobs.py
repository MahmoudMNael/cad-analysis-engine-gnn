from __future__ import annotations

from datetime import datetime

from pydantic import Field, HttpUrl, field_validator

from packages.contracts.dto.common import ApiModel, RequestModel, require_non_empty_id
from packages.contracts.enums import JobStatus, UnitSource


class CreateJobFromUrlRequest(RequestModel):
    url: HttpUrl
    unit: str | None = None
    consent_training: bool = False

    @field_validator("unit")
    @classmethod
    def _unit(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().lower()
        if not cleaned:
            raise ValueError("unit must be non-empty when provided")
        return cleaned


class JobDto(ApiModel):
    job_id: str
    status: JobStatus
    source_hash: str | None = None
    source_uri: str | None = None
    unit: str | None = None
    unit_source: UnitSource | None = None
    consent_training: bool = False
    engine_sha: str | None = None
    pipeline_version: str | None = None
    model_versions: dict[str, str] = Field(default_factory=dict)
    configuration_hash: str | None = None
    error_code: str | None = None
    error_detail: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    attempt: int = 0
    layout_rev: int = 0

    @field_validator("job_id")
    @classmethod
    def _id(cls, value: str) -> str:
        return require_non_empty_id(value)


class JobCreatedDto(ApiModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED

    @field_validator("job_id")
    @classmethod
    def _id(cls, value: str) -> str:
        return require_non_empty_id(value)


class JobCreatedResponse(ApiModel):
    success: bool = True
    data: JobCreatedDto
    message: str = "Job queued"


class JobResponse(ApiModel):
    success: bool = True
    data: JobDto
    message: str = ""
