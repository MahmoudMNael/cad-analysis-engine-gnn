from __future__ import annotations

from pathlib import Path

from packages.common.configuration import Settings
from packages.common.errors import AppError
from packages.common.hashing import configuration_hash, engine_sha_from_sources, sha256_file
from packages.common.logging import get_logger
from packages.common.mailer import notify_analysis_completed
from packages.contracts.enums import ErrorCode
from packages.store.db.repositories.jobs import JobRepository
from packages.store.db.repositories.models import ModelRegistryRepository
from packages.store.idempotency import snapshot_is_complete
from packages.store.models_registry import resolve_pinned_models
from packages.store.object_store import ObjectStore
from packages.store.snapshots import SnapshotStore
from packages.ingest.dwg_converter import convert_dwg_to_dxf, is_dwg
from apps.worker.pipeline.context import PipelineContext
from apps.worker.pipeline.runner import run_pipeline

log = get_logger("worker.jobs")
REPO_ROOT = Path(__file__).resolve().parents[3]


def _job_cancelled(jobs: JobRepository, job_id: str, extra_base: dict) -> bool:
    current = jobs.get(job_id)
    if current is not None and current.status == "cancelled":
        log.info("job_cancelled", extra={**extra_base, "stage": "cancel"})
        return True
    return False


def _notify_if_succeeded(jobs: JobRepository, job_id: str, settings: Settings) -> bool:
    current = jobs.get(job_id)
    if current is None or current.status != "succeeded":
        return False
    notify_analysis_completed(settings, job_id=job_id, notify_email=current.notify_email)
    return True


def process_job(
    job_id: str,
    *,
    settings: Settings,
    worker_id: str,
    conn,
    object_store: ObjectStore,
) -> None:
    jobs = JobRepository(conn)
    registry = ModelRegistryRepository(conn)
    snapshots = SnapshotStore(object_store, jobs)
    job = jobs.get(job_id)
    extra_base = {"job_id": job_id, "worker_id": worker_id}
    if job is None:
        log.error(
            "job_missing",
            extra={**extra_base, "stage": "load_job", "error_code": ErrorCode.JOB_NOT_FOUND.value},
        )
        return

    log.info("job_start", extra={**extra_base, "stage": "load_job"})

    artifacts = jobs.list_artifacts(job_id, job.snapshot_version or 1)
    if job.status == "succeeded" and snapshot_is_complete(job, object_store, artifacts):
        log.info("job_already_succeeded", extra={**extra_base, "stage": "idempotency"})
        return
    if job.status == "running" and snapshot_is_complete(job, object_store, artifacts):
        log.info("job_complete_from_snapshot", extra={**extra_base, "stage": "idempotency"})
        if _job_cancelled(jobs, job_id, extra_base):
            return
        meta = snapshots.load_meta(job_id)
        jobs.mark_succeeded(
            job_id,
            unit=str(meta.get("unit") or "m"),
            unit_source=str(meta.get("unit_source") or "fallback"),
            snapshot_version=1,
        )
        _notify_if_succeeded(jobs, job_id, settings)
        return
    if job.status == "failed":
        log.info("job_already_failed", extra={**extra_base, "stage": "load_job"})
        return
    if job.status == "cancelled":
        log.info("job_already_cancelled", extra={**extra_base, "stage": "load_job"})
        return
    if job.status == "queued":
        log.info("job_claim", extra={**extra_base, "stage": "claim"})
        claimed = jobs.claim_queued(job_id, worker_id)
        if claimed is None:
            log.info("job_claim_lost", extra={**extra_base, "stage": "claim"})
            return
        job = claimed
        log.info("job_running", extra={**extra_base, "stage": "claim"})
    elif job.status == "running":
        if job.worker_id not in {None, worker_id}:
            log.info(
                "job_owned_by_other_worker",
                extra={**extra_base, "stage": "claim", "error_detail": f"owner={job.worker_id}"},
            )
            return
        log.info("job_resume", extra={**extra_base, "stage": "claim"})
    else:
        log.error(
            "job_unexpected_status",
            extra={**extra_base, "stage": "claim", "error_detail": f"status={job.status}"},
        )
        return

    pinned = job.model_versions_json or resolve_pinned_models(registry).as_dict()
    face_version = str(pinned.get("face_model_version") or "heuristic:face_v1")
    face_checkpoint = registry.checkpoint_uri("face", face_version)
    engine_sha = job.engine_sha or settings.engine_sha_override or engine_sha_from_sources(REPO_ROOT)
    source_key = object_store.key_for_job(job_id, "source.dxf")
    if not object_store.exists(source_key):
        log.error(
            "source_missing",
            extra={
                **extra_base,
                "stage": "load_source",
                "error_code": ErrorCode.STORAGE_ERROR.value,
                "error_detail": "source artifact missing",
            },
        )
        jobs.mark_failed(job_id, ErrorCode.STORAGE_ERROR.value, "source artifact missing")
        return
    source_bytes = object_store.get_bytes(source_key)
    log.info(
        "source_loaded",
        extra={**extra_base, "stage": "load_source", "count_in": 1, "count_out": 1},
    )
    tmp_dir = Path(settings.object_store_dir) / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    original = getattr(job, "original_filename", None) or "upload.dxf"
    suffix = Path(original).suffix.lower() or ".dxf"
    tmp_source = tmp_dir / f"{job_id}{suffix}"
    tmp_source.write_bytes(source_bytes)
    pipeline_path = tmp_source
    tmp_dxf: Path | None = None
    context = PipelineContext(
        job_id=job_id,
        engine_sha=engine_sha,
        configuration=job.configuration_hash or configuration_hash(settings.job_config_dict()),
        model_versions=dict(pinned),
        source_hash=job.source_hash or sha256_file(tmp_source),
        worker_id=worker_id,
        model_checkpoint_uris={"face": face_checkpoint} if face_checkpoint else {},
    )
    try:
        if is_dwg(original, source_bytes[:512]):
            tmp_dxf = tmp_source.with_suffix(".dxf")
            convert_dwg_to_dxf(tmp_source, tmp_dxf, settings)
            pipeline_path = tmp_dxf
            log.info(
                "dwg_converted",
                extra={**extra_base, "stage": "dwg_convert", "count_in": 1, "count_out": 1},
            )
        result = run_pipeline(
            source_path=pipeline_path,
            context=context,
            settings=settings,
            explicit_unit=job.unit,
        )
        if _job_cancelled(jobs, job_id, extra_base):
            return
        with context.track_stage("snapshot", model_ref="heuristic:snapshot_v1", count_in=1) as stage:
            snapshots.write(
                job_id=job_id,
                source_bytes=source_bytes,
                document_payload=result.document_payload,
                plan_graph_payload=result.plan_graph_payload,
                faces_rows=result.faces_rows,
                layout=result.layout,
                meta=result.meta,
                snapshot_version=1,
            )
            stage.count_out = 6
        if _job_cancelled(jobs, job_id, extra_base):
            return
        with context.track_stage("persist_success", model_ref="heuristic:persist_v1", count_in=1) as stage:
            jobs.mark_succeeded(
                job_id,
                unit=result.document.unit,
                unit_source=result.document.unit_source.value,
                snapshot_version=1,
            )
            stage.count_out = 1
        if not _notify_if_succeeded(jobs, job_id, settings):
            return
        log.info(
            "job_succeeded",
            extra={
                **extra_base,
                "stage": "persist_success",
                "count_out": len(result.layout.physical_rooms),
            },
        )
    except AppError as exc:
        jobs.mark_failed(job_id, exc.code.value, exc.detail)
        log.exception(
            "job_failed",
            extra={
                **extra_base,
                "stage": context.current_stage or "unknown",
                "error_code": exc.code.value,
                "error_detail": exc.detail[:500],
            },
        )
    except Exception as exc:
        jobs.mark_failed(job_id, ErrorCode.INTERNAL_ERROR.value, str(exc))
        log.exception(
            "job_failed",
            extra={
                **extra_base,
                "stage": context.current_stage or "unknown",
                "error_code": ErrorCode.INTERNAL_ERROR.value,
                "error_detail": str(exc)[:500],
            },
        )
    finally:
        tmp_source.unlink(missing_ok=True)
        if tmp_dxf is not None:
            tmp_dxf.unlink(missing_ok=True)
