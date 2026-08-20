from __future__ import annotations

import time
import uuid

import psycopg

from packages.common.configuration import get_settings
from packages.common.errors import QueueError
from packages.common.logging import configure_logging, get_logger
from packages.contracts.enums import ErrorCode
from packages.store.db.migrate import migrate
from packages.store.db.repositories.jobs import JobRepository
from packages.store.jobs import JobQueue
from packages.store.object_store import ObjectStore
from apps.worker.jobs.analyze_job import process_job

log = get_logger("worker")


def _reclaim(conn, queue: JobQueue, timeout_seconds: int) -> None:
    jobs = JobRepository(conn)
    for stale in jobs.reset_stale_running(timeout_seconds):
        queue.enqueue(stale)
    for queued in jobs.list_queued():
        queue.enqueue(queued.id)


def main() -> None:
    configure_logging()
    settings = get_settings()
    migrate(settings)
    worker_id = f"worker-{uuid.uuid4()}"
    queue = JobQueue(settings.redis_url, settings.redis_queue_key)
    store = ObjectStore(settings.object_store_dir)
    log.info("worker_start", extra={"worker_id": worker_id})
    last_reclaim = 0.0
    while True:
        now = time.time()
        if now - last_reclaim > 15:
            try:
                with psycopg.connect(settings.database_url) as conn:
                    _reclaim(conn, queue, settings.job_timeout_seconds)
                last_reclaim = now
            except Exception:
                log.exception(
                    "worker_reclaim_failed",
                    extra={"worker_id": worker_id, "error_code": ErrorCode.QUEUE_ERROR.value},
                )
                time.sleep(1.0)
                continue
        try:
            job_id = queue.dequeue(timeout=5)
        except QueueError:
            log.exception(
                "queue_dequeue_failed",
                extra={"worker_id": worker_id, "error_code": ErrorCode.QUEUE_ERROR.value},
            )
            time.sleep(1.0)
            continue
        if not job_id:
            continue
        log.info("worker_job_dequeued", extra={"job_id": job_id, "worker_id": worker_id, "stage": "dequeue"})
        try:
            with psycopg.connect(settings.database_url) as conn:
                process_job(
                    job_id,
                    settings=settings,
                    worker_id=worker_id,
                    conn=conn,
                    object_store=store,
                )
        except Exception as exc:
            log.exception(
                "worker_unhandled",
                extra={
                    "job_id": job_id,
                    "worker_id": worker_id,
                    "stage": "process_job",
                    "error_code": ErrorCode.INTERNAL_ERROR.value,
                    "error_detail": str(exc)[:500],
                },
            )


if __name__ == "__main__":
    main()
