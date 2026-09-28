from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import integration_enabled
from tests.helpers.dxf_builder import add_rect, write_simple_dxf

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    if not integration_enabled():
        pytest.skip("CAD_RUN_INTEGRATION not set")
    from apps.api.app import create_app
    from packages.store.db.migrate import migrate

    migrate()
    return TestClient(create_app())


def _process(job_id: str) -> None:
    import psycopg

    from apps.worker.jobs.analyze_job import process_job
    from packages.common.configuration import get_settings
    from packages.store.object_store import ObjectStore

    settings = get_settings()
    with psycopg.connect(settings.database_url) as conn:
        process_job(
            job_id,
            settings=settings,
            worker_id="test-worker-p3",
            conn=conn,
            object_store=ObjectStore(settings.object_store_dir),
        )


def test_without_promoted_gnn_pipeline_uses_heuristic(client: TestClient, tmp_path: Path):
    dxf = write_simple_dxf(tmp_path / "p3.dxf", lambda msp: add_rect(msp, (0, 0), 5000, 4000))
    with dxf.open("rb") as handle:
        created = client.post(
            "/api/v1/jobs",
            files={"file": ("p3.dxf", handle, "application/dxf")},
            data={"consent_training": "true"},
        )
    assert created.status_code == 202, created.text
    job_id = created.json()["data"]["job_id"]
    _process(job_id)
    layout = client.get(f"/api/v1/jobs/{job_id}/layout")
    assert layout.status_code == 200, layout.text
    payload = layout.json()["data"]
    assert payload["physical_rooms"]
    assert payload["meta"]["model_versions"]["face_model_version"].startswith("heuristic:")
    assert payload["meta"]["model_decision_sources"]["face"] == "heuristic"
    models = client.get("/api/v1/models")
    assert models.status_code == 200
    names = {row["model_name"] for row in models.json()["data"]["models"]}
    assert "face" in names
