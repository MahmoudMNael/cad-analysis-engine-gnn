from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, File, Form, UploadFile

from apps.api.dependencies import (
    get_engine_sha,
    get_object_store,
    get_queue,
    job_repo,
    model_repo,
)
from packages.common.configuration import Settings, get_settings
from packages.common.errors import (
    FileTooLargeError,
    InvalidRequestError,
    JobFailedError,
    JobNotFoundError,
    JobNotReadyError,
    LayoutNotFoundError,
    UnsupportedFormatError,
)
from packages.common.hashing import configuration_hash, sha256_hex
from packages.contracts.dto.jobs import (
    CreateJobFromUrlRequest,
    JobCreatedDto,
    JobCreatedResponse,
    JobDto,
    JobResponse,
)
from packages.contracts.dto.layout import LayoutResponse
from packages.contracts.enums import JobStatus
from packages.ingest.dxf_loader import sniff_format
from packages.store.db.repositories.jobs import JobRepository
from packages.store.db.repositories.models import ModelRegistryRepository
from packages.store.jobs import JobQueue
from packages.store.models_registry import resolve_pinned_models
from packages.store.object_store import ObjectStore
from packages.store.snapshots import SnapshotStore
from apps.api.dependencies import snapshot_store

router = APIRouter()


def _sanitize_filename(name: str | None) -> str:
    raw = Path(name or "upload.dxf").name
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)
    return cleaned or "upload.dxf"


def _validate_upload(filename: str, data: bytes, settings: Settings) -> None:
    if len(data) > settings.max_upload_bytes:
        raise FileTooLargeError("uploaded file exceeds MAX_UPLOAD_BYTES")
    if not data:
        raise InvalidRequestError("empty upload")
    sniff_format(filename, data[:512])


def _to_job_dto(job, layout_rev: int = 0) -> JobDto:
    return JobDto(
        job_id=job.id,
        status=JobStatus(job.status),
        source_hash=job.source_hash,
        source_uri=job.source_uri,
        unit=job.unit,
        unit_source=job.unit_source,
        consent_training=job.consent_training,
        engine_sha=job.engine_sha,
        pipeline_version=job.pipeline_version,
        model_versions=dict(job.model_versions_json or {}),
        configuration_hash=job.configuration_hash,
        error_code=job.error_code,
        error_detail=job.error_detail,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        attempt=job.attempt,
        layout_rev=layout_rev,
    )


def _create_job(
    *,
    data: bytes,
    filename: str,
    consent_training: bool,
    unit: str | None,
    settings: Settings,
    jobs: JobRepository,
    models: ModelRegistryRepository,
    store: ObjectStore,
    queue: JobQueue,
    engine_sha: str,
) -> JobCreatedDto:
    _validate_upload(filename, data, settings)
    job_id = str(uuid.uuid4())
    source_hash = sha256_hex(data)
    pinned = resolve_pinned_models(models)
    config_hash = configuration_hash(settings.job_config_dict())
    key = store.key_for_job(job_id, "source.dxf")
    uri, _digest = store.put_bytes(key, data)
    jobs.insert_queued(
        job_id=job_id,
        source_hash=source_hash,
        source_uri=uri,
        consent_training=consent_training,
        original_filename=_sanitize_filename(filename),
        configuration_hash=config_hash,
        pipeline_version=settings.pipeline_version,
        engine_sha=engine_sha,
        model_versions=pinned.as_dict(),
        unit=unit,
        unit_source="explicit" if unit else None,
    )
    queue.enqueue(job_id)
    return JobCreatedDto(job_id=job_id, status=JobStatus.QUEUED)


@router.post("/jobs", response_model=JobCreatedResponse, status_code=202)
async def create_job(
    file: UploadFile = File(...),
    unit: str | None = Form(default=None),
    consent_training: bool = Form(default=False),
    settings: Settings = Depends(get_settings),
    jobs: JobRepository = Depends(job_repo),
    models: ModelRegistryRepository = Depends(model_repo),
    store: ObjectStore = Depends(get_object_store),
    queue: JobQueue = Depends(get_queue),
    engine_sha: str = Depends(get_engine_sha),
) -> JobCreatedResponse:
    data = await file.read()
    created = _create_job(
        data=data,
        filename=file.filename or "upload.dxf",
        consent_training=consent_training,
        unit=unit,
        settings=settings,
        jobs=jobs,
        models=models,
        store=store,
        queue=queue,
        engine_sha=engine_sha,
    )
    return JobCreatedResponse(success=True, data=created, message="Job queued")


@router.post("/jobs/from-url", response_model=JobCreatedResponse, status_code=202)
def create_job_from_url(
    body: CreateJobFromUrlRequest,
    settings: Settings = Depends(get_settings),
    jobs: JobRepository = Depends(job_repo),
    models: ModelRegistryRepository = Depends(model_repo),
    store: ObjectStore = Depends(get_object_store),
    queue: JobQueue = Depends(get_queue),
    engine_sha: str = Depends(get_engine_sha),
) -> JobCreatedResponse:
    parsed = urlparse(str(body.url))
    host = (parsed.hostname or "").lower()
    allowed = {h.lower() for h in settings.allowed_hosts()}
    if not allowed or host not in allowed:
        raise InvalidRequestError("URL host is not in the approved R2 allow-list", fields={"url": str(body.url)})
    if parsed.scheme != "https":
        raise InvalidRequestError("source URL must be https")
    try:
        response = httpx.get(str(body.url), follow_redirects=False, timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise InvalidRequestError(f"failed to download source URL: {exc}") from exc
    filename = Path(parsed.path).name or "remote.dxf"
    created = _create_job(
        data=response.content,
        filename=filename,
        consent_training=body.consent_training,
        unit=body.unit,
        settings=settings,
        jobs=jobs,
        models=models,
        store=store,
        queue=queue,
        engine_sha=engine_sha,
    )
    return JobCreatedResponse(success=True, data=created, message="Job queued")


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(
    job_id: str,
    jobs: JobRepository = Depends(job_repo),
) -> JobResponse:
    job = jobs.get(job_id)
    if job is None:
        raise JobNotFoundError(job_id)
    return JobResponse(success=True, data=_to_job_dto(job, jobs.layout_rev(job_id)), message="")


@router.get("/jobs/{job_id}/layout", response_model=LayoutResponse)
def get_layout(
    job_id: str,
    jobs: JobRepository = Depends(job_repo),
    snaps: SnapshotStore = Depends(snapshot_store),
) -> LayoutResponse:
    job = jobs.get(job_id)
    if job is None:
        raise JobNotFoundError(job_id)
    if job.status == JobStatus.FAILED.value:
        raise JobFailedError(job_id, job.error_detail or job.error_code or "failed")
    if job.status != JobStatus.SUCCEEDED.value:
        raise JobNotReadyError(job_id, job.status)
    try:
        layout = snaps.load_layout(job_id)
    except Exception as exc:
        raise LayoutNotFoundError(job_id) from exc
    return LayoutResponse(success=True, data=layout, message="")
