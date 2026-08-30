from __future__ import annotations

import os
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


def test_health_reports_dependencies(client: TestClient):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert "postgres" in body["data"]
    assert "redis" in body["data"]
    assert "object_store" in body["data"]


def test_docs_and_openapi(client: TestClient):
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200
    spec = client.get("/openapi.json")
    assert spec.status_code == 200
    assert spec.json()["paths"]


def test_upload_queue_process_layout(client: TestClient, tmp_path: Path):
    from packages.common.configuration import get_settings
    from packages.store.object_store import ObjectStore
    from packages.store.jobs import JobQueue
    import psycopg
    from apps.worker.jobs.analyze_job import process_job

    dxf = write_simple_dxf(tmp_path / "job.dxf", lambda msp: add_rect(msp, (0, 0), 5000, 4000))
    with dxf.open("rb") as handle:
        res = client.post(
            "/api/v1/jobs",
            files={"file": ("job.dxf", handle, "application/dxf")},
            data={"consent_training": "false"},
        )
    assert res.status_code == 202, res.text
    job_id = res.json()["data"]["job_id"]
    status = client.get(f"/api/v1/jobs/{job_id}")
    assert status.json()["data"]["status"] == "queued"

    settings = get_settings()
    with psycopg.connect(settings.database_url) as conn:
        process_job(
            job_id,
            settings=settings,
            worker_id="test-worker",
            conn=conn,
            object_store=ObjectStore(settings.object_store_dir),
        )

    done = client.get(f"/api/v1/jobs/{job_id}")
    assert done.json()["data"]["status"] == "succeeded"
    layout = client.get(f"/api/v1/jobs/{job_id}/layout")
    assert layout.status_code == 200, layout.text
    payload = layout.json()["data"]
    assert payload["unit"] == "m"
    assert payload["physical_rooms"]
    store = ObjectStore(settings.object_store_dir)
    assert store.snapshot_complete(job_id)

    # idempotent replay must not duplicate snapshot kinds
    with psycopg.connect(settings.database_url) as conn:
        process_job(
            job_id,
            settings=settings,
            worker_id="test-worker",
            conn=conn,
            object_store=store,
        )
    again = client.get(f"/api/v1/jobs/{job_id}")
    assert again.json()["data"]["status"] == "succeeded"


def test_unsupported_format(client: TestClient):
    res = client.post(
        "/api/v1/jobs",
        files={"file": ("plan.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "UNSUPPORTED_FORMAT"


def test_dwg_upload_accepted(client: TestClient):
    res = client.post(
        "/api/v1/jobs",
        files={"file": ("plan.dwg", b"AC1027xxxx", "application/octet-stream")},
    )
    assert res.status_code == 202, res.text
    assert res.json()["data"]["status"] == "queued"


def test_layout_not_ready(client: TestClient, tmp_path: Path):
    dxf = write_simple_dxf(tmp_path / "wait.dxf", lambda msp: add_rect(msp, (0, 0), 3000, 3000))
    with dxf.open("rb") as handle:
        res = client.post("/api/v1/jobs", files={"file": ("wait.dxf", handle, "application/dxf")})
    job_id = res.json()["data"]["job_id"]
    layout = client.get(f"/api/v1/jobs/{job_id}/layout")
    assert layout.status_code == 409
    assert layout.json()["error"]["code"] == "JOB_NOT_READY"


def test_explicit_unit_reaches_layout_meta(client: TestClient, tmp_path: Path):
    from ezdxf.enums import InsertUnits

    from packages.common.configuration import get_settings
    from packages.store.object_store import ObjectStore
    import psycopg
    from apps.worker.jobs.analyze_job import process_job

    dxf = write_simple_dxf(
        tmp_path / "inch.dxf",
        lambda msp: add_rect(msp, (0, 0), 5, 4),
        insunits=InsertUnits.Inches,
    )
    with dxf.open("rb") as handle:
        res = client.post(
            "/api/v1/jobs",
            files={"file": ("inch.dxf", handle, "application/dxf")},
            data={"unit": "m", "consent_training": "false"},
        )
    assert res.status_code == 202, res.text
    job_id = res.json()["data"]["job_id"]
    settings = get_settings()
    with psycopg.connect(settings.database_url) as conn:
        process_job(
            job_id,
            settings=settings,
            worker_id="test-worker",
            conn=conn,
            object_store=ObjectStore(settings.object_store_dir),
        )
    layout = client.get(f"/api/v1/jobs/{job_id}/layout")
    assert layout.status_code == 200, layout.text
    meta = layout.json()["data"]["meta"]
    assert meta["unit_source"] == "explicit"
    assert abs(meta["scale_factor"] - 1.0) < 1e-12
