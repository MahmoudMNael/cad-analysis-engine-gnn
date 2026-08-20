from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends

from apps.api.dependencies import get_object_store, get_queue
from packages.common.configuration import Settings, get_settings
from packages.contracts.dto.health import ComponentHealthDto, HealthDataDto, HealthResponse
from packages.store.jobs import JobQueue
from packages.store.object_store import ObjectStore

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(
    settings: Settings = Depends(get_settings),
    queue: JobQueue = Depends(get_queue),
    store: ObjectStore = Depends(get_object_store),
) -> HealthResponse:
    pg_ok = True
    pg_detail = None
    try:
        with psycopg.connect(settings.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
    except Exception as exc:
        pg_ok = False
        pg_detail = str(exc)

    redis_ok = queue.ping()
    store_ok = store.healthcheck()
    statuses = [pg_ok, redis_ok, store_ok]
    if all(statuses):
        overall = "ok"
    elif any(statuses):
        overall = "degraded"
    else:
        overall = "error"
    data = HealthDataDto(
        status=overall,
        postgres=ComponentHealthDto(status="ok" if pg_ok else "error", detail=pg_detail),
        redis=ComponentHealthDto(status="ok" if redis_ok else "error", detail=None if redis_ok else "ping failed"),
        object_store=ComponentHealthDto(status="ok" if store_ok else "error", detail=None if store_ok else "write failed"),
    )
    return HealthResponse(success=True, data=data, message="health")
