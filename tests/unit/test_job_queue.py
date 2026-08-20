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
