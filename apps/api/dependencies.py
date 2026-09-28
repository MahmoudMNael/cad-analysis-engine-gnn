from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import psycopg
from fastapi import Depends

from packages.common.configuration import Settings, get_settings
from packages.common.hashing import engine_sha_from_sources
from packages.store.db.repositories.jobs import JobRepository
from packages.store.db.repositories.models import ModelRegistryRepository
from packages.store.events import EventStore
from packages.store.jobs import JobQueue
from packages.store.object_store import ObjectStore
from packages.store.snapshots import SnapshotStore

REPO_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def get_object_store() -> ObjectStore:
    return ObjectStore(get_settings().object_store_dir)


@lru_cache(maxsize=1)
def get_queue() -> JobQueue:
    settings = get_settings()
    return JobQueue(settings.redis_url, settings.redis_queue_key)


def get_engine_sha(settings: Settings = Depends(get_settings)) -> str:
    return settings.engine_sha_override or engine_sha_from_sources(REPO_ROOT)


def db_conn(settings: Settings = Depends(get_settings)):
    conn = psycopg.connect(settings.database_url)
    try:
        yield conn
    finally:
        conn.close()


def job_repo(conn=Depends(db_conn)) -> JobRepository:
    return JobRepository(conn)


def model_repo(conn=Depends(db_conn)) -> ModelRegistryRepository:
    return ModelRegistryRepository(conn)


def snapshot_store(
    jobs: JobRepository = Depends(job_repo),
    objects: ObjectStore = Depends(get_object_store),
) -> SnapshotStore:
    return SnapshotStore(objects, jobs)


def event_store(conn=Depends(db_conn)) -> EventStore:
    return EventStore(conn)
