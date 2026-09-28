from __future__ import annotations

import json
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
            worker_id="test-worker-notebook",
            conn=conn,
            object_store=ObjectStore(settings.object_store_dir),
        )


def _checkpoint_bytes() -> bytes:
    try:
        from packages.models_nn.face_gnn import FaceGNN

        return FaceGNN(hidden_dim=8, num_layers=1, dropout=0.0).save_bytes()
    except ImportError:
        return b"placeholder-checkpoint"


def _assert_no_cad_payload(data: dict) -> None:
    dumped = json.dumps(data)
    assert "wkt" not in dumped
    assert ".dxf" not in dumped.lower()
    for graph in data.get("graphs") or []:
        assert "wkt" not in graph
        assert "dxf" not in graph


def test_export_bundle_has_eligibility_and_no_cad_payloads(client: TestClient):
    response = client.get("/api/v1/admin/datasets/face")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["success"] is True
    data = payload["data"]
    assert "ok" in data["eligibility"]
    assert "labeled_jobs" in data["eligibility"]
    assert "reasons" in data["eligibility"]
    assert data["feature_schema_version"] == "face_features_v1"
    _assert_no_cad_payload(data)


def test_export_skips_jobs_without_training_consent(client: TestClient, tmp_path: Path):
    dxf = write_simple_dxf(tmp_path / "nocon.dxf", lambda msp: add_rect(msp, (0, 0), 5000, 4000))
    with dxf.open("rb") as handle:
        created = client.post(
            "/api/v1/jobs",
            files={"file": ("nocon.dxf", handle, "application/dxf")},
            data={"consent_training": "false"},
        )
    assert created.status_code == 202, created.text
    job_id = created.json()["data"]["job_id"]
    _process(job_id)
    layout = client.get(f"/api/v1/jobs/{job_id}/layout")
    assert layout.status_code == 200, layout.text
    rooms = layout.json()["data"]["rooms"] or layout.json()["data"]["physical_rooms"]
    assert rooms
    rev = layout.json()["data"]["meta"]["layout_rev"]
    deleted = client.post(
        f"/api/v1/jobs/{job_id}/events",
        json={
            "action": "delete_room",
            "target_ids": [rooms[0]["id"]],
            "payload": {"label": "not_indoor"},
            "expected_layout_rev": rev,
            "actor": "user",
        },
    )
    assert deleted.status_code == 200, deleted.text
    exported = client.get("/api/v1/admin/datasets/face")
    assert exported.status_code == 200, exported.text
    data = exported.json()["data"]
    _assert_no_cad_payload(data)
    job_ids = {graph["job_id"] for graph in data["graphs"]}
    assert job_id not in job_ids


def test_export_after_consented_job_has_numeric_graphs(client: TestClient, tmp_path: Path):
    dxf = write_simple_dxf(tmp_path / "nb.dxf", lambda msp: add_rect(msp, (0, 0), 5000, 4000))
    with dxf.open("rb") as handle:
        created = client.post(
            "/api/v1/jobs",
            files={"file": ("nb.dxf", handle, "application/dxf")},
            data={"consent_training": "true"},
        )
    assert created.status_code == 202, created.text
    job_id = created.json()["data"]["job_id"]
    _process(job_id)
    layout = client.get(f"/api/v1/jobs/{job_id}/layout")
    assert layout.status_code == 200, layout.text
    rooms = layout.json()["data"]["rooms"] or layout.json()["data"]["physical_rooms"]
    assert rooms
    rev = layout.json()["data"]["meta"]["layout_rev"]
    deleted = client.post(
        f"/api/v1/jobs/{job_id}/events",
        json={
            "action": "delete_room",
            "target_ids": [rooms[0]["id"]],
            "payload": {"label": "not_indoor"},
            "expected_layout_rev": rev,
            "actor": "user",
        },
    )
    assert deleted.status_code == 200, deleted.text
    exported = client.get("/api/v1/admin/datasets/face")
    assert exported.status_code == 200, exported.text
    data = exported.json()["data"]
    _assert_no_cad_payload(data)
    assert data["eligibility"]["labeled_jobs"] >= 1
    graph = next(item for item in data["graphs"] if item["job_id"] == job_id)
    assert graph["node_features"]
    assert graph["labels"]
    assert all(isinstance(row, list) for row in graph["node_features"])
    assert all(isinstance(value, (int, float)) for row in graph["node_features"] for value in row)


def test_import_stores_candidate_without_promoting(client: TestClient, tmp_path: Path):
    metrics = {
        "feature_schema_version": "face_features_v1",
        "macro_f1": 0.91,
        "heuristic_macro_f1": 0.4,
        "heuristic_f1_indoor": 0.4,
        "heuristic_delete_rate_proxy": 0.2,
        "held_out_passed": 1.0,
    }
    files = {
        "checkpoint": ("face_gnn.pt", _checkpoint_bytes(), "application/octet-stream"),
        "metrics": ("metrics.json", json.dumps(metrics).encode("utf-8"), "application/json"),
    }
    imported = client.post("/api/v1/admin/models", files=files)
    assert imported.status_code == 200, imported.text
    body = imported.json()["data"]
    assert body["status"] == "candidate"
    assert body["model_name"] == "face"
    assert body["version"].startswith("gnn:face:")
    assert body["feature_schema_version"] == "face_features_v1"

    listed = client.get("/api/v1/models")
    assert listed.status_code == 200, listed.text
    versions = {row["version"]: row for row in listed.json()["data"]["models"]}
    assert body["version"] in versions
    assert versions[body["version"]]["status"] == "candidate"

    dxf = write_simple_dxf(tmp_path / "pinned.dxf", lambda msp: add_rect(msp, (0, 0), 4000, 3000))
    with dxf.open("rb") as handle:
        created = client.post(
            "/api/v1/jobs",
            files={"file": ("pinned.dxf", handle, "application/dxf")},
            data={"consent_training": "true"},
        )
    assert created.status_code == 202, created.text
    job_id = created.json()["data"]["job_id"]
    _process(job_id)
    layout = client.get(f"/api/v1/jobs/{job_id}/layout")
    assert layout.status_code == 200, layout.text
    face_version = layout.json()["data"]["meta"]["model_versions"]["face_model_version"]
    assert face_version.startswith("heuristic:")
    assert layout.json()["data"]["meta"]["model_decision_sources"]["face"] == "heuristic"


def test_admin_train_still_unavailable(client: TestClient):
    response = client.post("/api/v1/admin/train", json={"model_name": "face", "force": False})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "MODEL_NOT_AVAILABLE"
