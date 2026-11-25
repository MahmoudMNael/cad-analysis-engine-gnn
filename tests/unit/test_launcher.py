from __future__ import annotations

from apps import launcher


class _FakeProc:
    def __init__(self, cmd: list[str], env: dict[str, str], exit_after_polls: int | None = None, rc: int = 1):
        self.cmd = cmd
        self.env = env
        self._polls = 0
        self._exit_after_polls = exit_after_polls
        self.returncode: int | None = None
        self._exit_rc = rc
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        if self.terminated or self.killed:
            if self.returncode is None:
                self.returncode = 0
            return self.returncode
        self._polls += 1
        if self._exit_after_polls is not None and self._polls >= self._exit_after_polls:
            self.returncode = self._exit_rc
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        if self.returncode is None:
            self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode or 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def test_supervise_spawns_isolated_workers_and_respawns(monkeypatch, tmp_path):
    spawned: list[_FakeProc] = []
    now = {"t": 0.0}
    loops = {"n": 0}

    def fake_popen(cmd, env):
        is_worker = "apps.worker.main" in cmd
        worker_spawns = sum(1 for p in spawned if "apps.worker.main" in p.cmd)
        exit_after = 1 if (is_worker and worker_spawns == 0) else None
        proc = _FakeProc(cmd, env, exit_after_polls=exit_after, rc=137)
        spawned.append(proc)
        return proc

    def fake_sleep(dt: float) -> None:
        now["t"] += dt
        loops["n"] += 1

    monkeypatch.setattr(launcher.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(launcher.time, "monotonic", lambda: now["t"])
    monkeypatch.setattr(launcher.time, "sleep", fake_sleep)

    rc = launcher.supervise(
        port=8000,
        worker_count=2,
        base_env={"PATH": "/usr/bin"},
        should_stop=lambda: loops["n"] >= 6,
        respawn_delay=1.0,
        poll_interval=0.5,
    )
    assert rc == 0
    assert "uvicorn" in spawned[0].cmd
    assert spawned[0].cmd[-1] == "8000"
    workers = [p for p in spawned if "apps.worker.main" in p.cmd]
    # 2 initial workers + 1 respawned after worker 0 crashed
    assert len(workers) == 3
    assert workers[0].env["XDG_RUNTIME_DIR"] == str(tmp_path / "runtime-worker-0")
    assert workers[0].env["CAD_XVFB_SERVER_NUM"] == "99"
    assert workers[1].env["XDG_RUNTIME_DIR"] == str(tmp_path / "runtime-worker-1")
    assert workers[1].env["CAD_XVFB_SERVER_NUM"] == "100"
    assert workers[2].env["XDG_RUNTIME_DIR"] == str(tmp_path / "runtime-worker-0")
    assert workers[2].env["CAD_XVFB_SERVER_NUM"] == "99"


def test_supervise_exits_nonzero_if_api_dies(monkeypatch, tmp_path):
    spawned: list[_FakeProc] = []

    def fake_popen(cmd, env):
        is_api = "uvicorn" in cmd
        proc = _FakeProc(cmd, env, exit_after_polls=1 if is_api else None, rc=2)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(launcher.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(launcher.time, "sleep", lambda _dt: None)

    rc = launcher.supervise(port=8000, worker_count=2, base_env={})
    assert rc == 2
    assert spawned[1].terminated is True
    assert spawned[2].terminated is True
