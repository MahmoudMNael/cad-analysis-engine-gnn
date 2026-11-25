from __future__ import annotations

import redis
from redis.exceptions import TimeoutError as RedisTimeoutError

from packages.common.errors import QueueError
from packages.store.jobs import JobQueue


def _queue_with_client(client: object) -> JobQueue:
    queue = JobQueue.__new__(JobQueue)
    queue._key = "cad:jobs"
    queue._client = client
    return queue


class _IdleClient:
    def blpop(self, key: str, timeout: int = 5):
        raise RedisTimeoutError("Timeout reading from socket")


class _BrokenClient:
    def blpop(self, key: str, timeout: int = 5):
        raise redis.ConnectionError("redis down")


class _EmptyClient:
    def blpop(self, key: str, timeout: int = 5):
        return None


def test_dequeue_idle_socket_timeout_returns_none():
    assert _queue_with_client(_IdleClient()).dequeue(timeout=5) is None


def test_dequeue_empty_blpop_returns_none():
    assert _queue_with_client(_EmptyClient()).dequeue(timeout=5) is None


def test_dequeue_connection_error_is_queue_error():
    try:
        _queue_with_client(_BrokenClient()).dequeue(timeout=5)
        raise AssertionError("expected QueueError")
    except QueueError as exc:
        assert "failed to dequeue" in str(exc)


class _RemoveClient:
    def __init__(self):
        self.calls = []

    def lrem(self, key: str, count: int, value: str):
        self.calls.append((key, count, value))
        return 1


class _BrokenRemoveClient:
    def lrem(self, key: str, count: int, value: str):
        raise redis.ConnectionError("redis down")


def test_remove_lrems_job_id():
    client = _RemoveClient()
    _queue_with_client(client).remove("job-1")
    assert client.calls == [("cad:jobs", 0, "job-1")]


def test_remove_connection_error_is_queue_error():
    try:
        _queue_with_client(_BrokenRemoveClient()).remove("job-1")
        raise AssertionError("expected QueueError")
    except QueueError as exc:
        assert "failed to remove" in str(exc)


class _EnqueueClient:
    def __init__(self):
        self.queued: list[str] = []
        self.eval_calls = []

    def eval(self, script: str, numkeys: int, key: str, job_id: str):
        self.eval_calls.append((numkeys, key, job_id, "LPOS" in script))
        if job_id in self.queued:
            return 0
        self.queued.append(job_id)
        return 1


class _BrokenEvalClient:
    def eval(self, script: str, numkeys: int, key: str, job_id: str):
        raise redis.ConnectionError("redis down")


def test_enqueue_deduplicates_job_id():
    client = _EnqueueClient()
    queue = _queue_with_client(client)
    assert queue.enqueue("job-1") is True
    assert queue.enqueue("job-1") is False
    assert client.queued == ["job-1"]
    assert client.eval_calls[0][:3] == (1, "cad:jobs", "job-1")
    assert client.eval_calls[0][3] is True


def test_enqueue_connection_error_is_queue_error():
    try:
        _queue_with_client(_BrokenEvalClient()).enqueue("job-1")
        raise AssertionError("expected QueueError")
    except QueueError as exc:
        assert "failed to enqueue" in str(exc)


class _LockClient:
    def __init__(self, result: bool):
        self.result = result
        self.calls = []

    def set(self, key: str, value: str, nx: bool = False, ex: int | None = None):
        self.calls.append((key, value, nx, ex))
        return self.result


def test_acquire_reclaim_lock():
    client = _LockClient(True)
    assert _queue_with_client(client).acquire_reclaim_lock("w-1", ttl_seconds=15) is True
    assert client.calls == [("cad:jobs:reclaim_lock", "w-1", True, 15)]


def test_worker_reclaim_skips_duplicate_stale_and_filters_age(monkeypatch):
    from types import SimpleNamespace
    from apps.worker.main import _reclaim

    class _FakeRepo:
        def __init__(self, _conn):
            self.age_arg = None

        def reset_stale_running(self, timeout_seconds: int):
            assert timeout_seconds == 300
            return ["job-stale"]

        def list_queued(self, min_age_seconds: int = 0):
            self.age_arg = min_age_seconds
            return [SimpleNamespace(id="job-stale"), SimpleNamespace(id="job-old")]

    repo_holder = []

    def _make_repo(conn):
        r = _FakeRepo(conn)
        repo_holder.append(r)
        return r

    monkeypatch.setattr("apps.worker.main.JobRepository", _make_repo)
    client = _EnqueueClient()
    _reclaim(object(), _queue_with_client(client), 300)
    assert repo_holder[0].age_arg == 30
    assert client.queued == ["job-stale", "job-old"]
    assert len(client.eval_calls) == 2

