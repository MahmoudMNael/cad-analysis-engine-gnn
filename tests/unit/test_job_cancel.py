from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from packages.common.configuration import Settings
from apps.worker.jobs.analyze_job import process_job


def _patch_repos(monkeypatch, repo, snaps=None, registry=None):
    monkeypatch.setattr("apps.worker.jobs.analyze_job.JobRepository", lambda conn: repo)
    monkeypatch.setattr(
        "apps.worker.jobs.analyze_job.ModelRegistryRepository",
        lambda conn: registry or MagicMock(),
    )
    monkeypatch.setattr(
        "apps.worker.jobs.analyze_job.SnapshotStore",
        lambda *args, **kwargs: snaps or MagicMock(),
    )


def test_process_job_skips_cancelled(monkeypatch):
    job = SimpleNamespace(status="cancelled", snapshot_version=1, notify_email="a@b.com")
    repo = MagicMock()
    repo.get.return_value = job
    repo.list_artifacts.return_value = {}
    _patch_repos(monkeypatch, repo)
    process_job(
        "job-1",
        settings=Settings(),
        worker_id="w",
        conn=object(),
        object_store=MagicMock(),
    )
    repo.mark_succeeded.assert_not_called()
    repo.claim_queued.assert_not_called()


def test_process_job_aborts_cancelled_before_snapshot(monkeypatch, tmp_path):
    running = SimpleNamespace(
        status="running",
        snapshot_version=1,
        notify_email="a@b.com",
        worker_id="w1",
        model_versions_json={},
        engine_sha="sha",
        configuration_hash="cfg",
        source_hash="a" * 64,
        unit=None,
    )
    cancelled = SimpleNamespace(status="cancelled", notify_email="a@b.com")
    repo = MagicMock()
    repo.get.side_effect = [running, cancelled]
    repo.list_artifacts.return_value = {}
    snaps = MagicMock()
    store = MagicMock()
    store.exists.return_value = True
    store.get_bytes.return_value = b"dxf"
    store.key_for_job.return_value = "jobs/job-1/source.dxf"
    registry = MagicMock()
    registry.checkpoint_uri.return_value = None
    _patch_repos(monkeypatch, repo, snaps=snaps, registry=registry)
    monkeypatch.setattr("apps.worker.jobs.analyze_job.snapshot_is_complete", lambda *a, **k: False)
    monkeypatch.setattr("apps.worker.jobs.analyze_job.run_pipeline", lambda **k: MagicMock())
    process_job(
        "job-1",
        settings=Settings(object_store_dir=tmp_path),
        worker_id="w1",
        conn=object(),
        object_store=store,
    )
    snaps.write.assert_not_called()
    repo.mark_succeeded.assert_not_called()
