"""Redis job queue plus Postgres-backed reclaim of queued work."""

from __future__ import annotations

import redis
from redis.exceptions import TimeoutError as RedisTimeoutError

from packages.common.errors import QueueError

# Must exceed BLPOP wait so an empty queue is a nil reply, not a client socket timeout.
_SOCKET_TIMEOUT_SECONDS = 30.0
_CONNECT_TIMEOUT_SECONDS = 5.0
_ENQUEUE_UNIQUE_LUA = """
if redis.call('LPOS', KEYS[1], ARGV[1]) == false then
    redis.call('RPUSH', KEYS[1], ARGV[1])
    return 1
end
return 0
"""


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

    def enqueue(self, job_id: str) -> bool:
        """Push job_id to the queue if not already queued."""
        try:
            return bool(self._client.eval(_ENQUEUE_UNIQUE_LUA, 1, self._key, job_id))
        except redis.RedisError as exc:
            raise QueueError(f"failed to enqueue job: {exc}") from exc

    def acquire_reclaim_lock(self, owner: str, ttl_seconds: int = 15) -> bool:
        """Acquire short-lived distributed lock so only one worker runs reclaim per tick."""
        try:
            return bool(self._client.set(f"{self._key}:reclaim_lock", owner, nx=True, ex=ttl_seconds))
        except redis.RedisError as exc:
            raise QueueError(f"failed to acquire reclaim lock: {exc}") from exc

    def remove(self, job_id: str) -> None:
        try:
            self._client.lrem(self._key, 0, job_id)
        except redis.RedisError as exc:
            raise QueueError(f"failed to remove job: {exc}") from exc

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
