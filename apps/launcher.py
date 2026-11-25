"""Single-container launcher running migrations, uvicorn, and N worker processes."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from packages.common.configuration import Settings, get_settings
from packages.common.logging import configure_logging, get_logger
from packages.store.db.migrate import migrate

log = get_logger("launcher")


def _migrate_with_retry(settings: Settings, *, attempts: int = 15, delay_seconds: float = 2.0) -> None:
    for attempt in range(1, attempts + 1):
        try:
            migrate(settings)
            return
        except Exception as exc:
            if attempt >= attempts:
                raise
            log.warning(
                "launcher_migrate_retry",
                extra={"stage": "migrate", "error_detail": str(exc)[:300], "attempt": attempt},
            )
            time.sleep(delay_seconds)


def _worker_env(base_env: Mapping[str, str], slot: int) -> dict[str, str]:
    runtime_dir = Path(tempfile.gettempdir()) / f"runtime-worker-{slot}"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.chmod(0o700)
    env = dict(base_env)
    env["XDG_RUNTIME_DIR"] = str(runtime_dir)
    env["CAD_XVFB_SERVER_NUM"] = str(99 + slot)
    return env


def _spawn_api(port: int, base_env: Mapping[str, str]) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", str(port)],
        env=dict(base_env),
    )


def _spawn_worker(slot: int, base_env: Mapping[str, str]) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "apps.worker.main"],
        env=_worker_env(base_env, slot),
    )


def _shutdown_children(procs: Sequence[subprocess.Popen], timeout_seconds: float = 10.0) -> None:
    alive = [proc for proc in procs if proc.poll() is None]
    for proc in alive:
        proc.terminate()
    deadline = time.monotonic() + timeout_seconds
    for proc in alive:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def supervise(
    *,
    port: int,
    worker_count: int,
    base_env: Mapping[str, str] | None = None,
    should_stop: Callable[[], bool] | None = None,
    respawn_delay: float = 2.0,
    poll_interval: float = 0.5,
) -> int:
    """Supervise one uvicorn API process and N isolated worker processes."""
    env = dict(os.environ if base_env is None else base_env)
    count = max(1, worker_count)
    stop_fn = should_stop or (lambda: False)

    api = _spawn_api(port, env)
    workers: list[subprocess.Popen | None] = [_spawn_worker(i, env) for i in range(count)]
    respawn_at: list[float | None] = [None] * count
    log.info("launcher_started", extra={"port": port, "worker_count": count})

    try:
        while not stop_fn():
            api_rc = api.poll()
            if api_rc is not None:
                log.error("launcher_api_exited", extra={"returncode": api_rc})
                return api_rc or 1

            now = time.monotonic()
            for idx in range(count):
                proc = workers[idx]
                if proc is not None:
                    rc = proc.poll()
                    if rc is not None:
                        log.warning(
                            "launcher_worker_exited",
                            extra={"worker_slot": idx, "returncode": rc},
                        )
                        workers[idx] = None
                        # ponytail: 2s fixed backoff without crash-loop circuit breaker; API stays up and _reclaim recovers the interrupted job.
                        respawn_at[idx] = now + respawn_delay
                due = respawn_at[idx]
                if workers[idx] is None and due is not None and now >= due:
                    log.info("launcher_worker_respawn", extra={"worker_slot": idx})
                    workers[idx] = _spawn_worker(idx, env)
                    respawn_at[idx] = None

            time.sleep(poll_interval)
        return 0
    finally:
        _shutdown_children([api, *(w for w in workers if w is not None)])


def main() -> None:
    configure_logging()
    settings = get_settings()
    _migrate_with_retry(settings)
    stopping = False

    def _handle_signal(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    port = int(os.environ.get("PORT", "8000"))
    raise SystemExit(
        supervise(
            port=port,
            worker_count=settings.worker_count,
            should_stop=lambda: stopping,
        )
    )


if __name__ == "__main__":
    main()
