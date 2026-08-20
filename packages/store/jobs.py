"""Redis job queue plus Postgres-backed reclaim of queued work."""

from __future__ import annotations

import redis
from redis.exceptions import TimeoutError as RedisTimeoutError

from packages.common.errors import QueueError

# Must exceed BLPOP wait so an empty queue is a nil reply, not a client socket timeout.
_SOCKET_TIMEOUT_SECONDS = 30.0
_CONNECT_TIMEOUT_SECONDS = 5.0


class JobQueue:
    def __init__(self, redis_url: str, key: str) -> None:
        self._key = key
        self._client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=_CONNECT_TIMEOUT_SECONDS,
            socket_timeout=_SOCKET_TIMEOUT_SECONDS,
            retry_on_timeout=False,
        )

    def enqueue(self, job_id: str) -> None:
        try:
            self._client.rpush(self._key, job_id)
        except redis.RedisError as exc:
            raise QueueError(f"failed to enqueue job: {exc}") from exc

    def dequeue(self, timeout: int = 5) -> str | None:
        if timeout >= _SOCKET_TIMEOUT_SECONDS:
            raise ValueError("dequeue timeout must be less than Redis socket timeout")
        try:
            item = self._client.blpop(self._key, timeout=timeout)
        except (RedisTimeoutError, TimeoutError):
            return None
        except redis.RedisError as exc:
            raise QueueError(f"failed to dequeue job: {exc}") from exc
        if not item:
            return None
        return item[1]

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except redis.RedisError:
            return False
