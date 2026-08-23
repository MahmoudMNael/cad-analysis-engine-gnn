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
            worker_id="test-worker-p2",
            conn=conn,
            object_store=ObjectStore(settings.object_store_dir),
        )


def test_feedback_event_revision_derived_layout_and_compile(client: TestClient, tmp_path: Path):
    import psycopg

    from apps.trainer.compile import compile_lake
    from packages.common.configuration import get_settings
    from packages.store.object_store import ObjectStore

    dxf = write_simple_dxf(tmp_path / "p2.dxf", lambda msp: add_rect(msp, (0, 0), 5000, 4000))
    with dxf.open("rb") as handle:
        created = client.post(
            "/api/v1/jobs",
            files={"file": ("p2.dxf", handle, "application/dxf")},
            data={"consent_training": "true"},
        )
    assert created.status_code == 202, created.text
    job_id = created.json()["data"]["job_id"]
    _process(job_id)

    layout = client.get(f"/api/v1/jobs/{job_id}/layout")
    assert layout.status_code == 200, layout.text
    rooms = layout.json()["data"]["rooms"]
    physical = layout.json()["data"]["physical_rooms"]
    assert rooms
    assert physical
    room_id = rooms[0]["id"]
    assert layout.json()["data"]["meta"]["layout_rev"] == 0

    faces = client.get(f"/api/v1/jobs/{job_id}/faces")
    assert faces.status_code == 200, faces.text
    assert faces.json()["data"]["faces"]
    assert any(face["id"] == room_id for face in faces.json()["data"]["faces"])

    deleted = client.post(
        f"/api/v1/jobs/{job_id}/events",
        json={
            "action": "delete_room",
            "target_ids": [room_id],
            "expected_layout_rev": 0,
            "actor": "user",
        },
    )
    assert deleted.status_code == 200, deleted.text
    event = deleted.json()["data"]
    assert event["layout_rev"] == 1
    assert event["sequence"] == 1
    assert event["action"] == "delete_room"

    after_delete = client.get(f"/api/v1/jobs/{job_id}/layout").json()["data"]
    assert after_delete["rooms"] == []
    assert after_delete["physical_rooms"]
    assert after_delete["meta"]["layout_rev"] == 1

    conflict = client.post(
        f"/api/v1/jobs/{job_id}/events",
        json={
            "action": "accept",
            "expected_layout_rev": 0,
        },
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "LAYOUT_REVISION_CONFLICT"

    undone = client.post(
        f"/api/v1/jobs/{job_id}/events",
        json={"action": "undo", "expected_layout_rev": 1},
    )
    assert undone.status_code == 200, undone.text
    restored = client.get(f"/api/v1/jobs/{job_id}/layout").json()["data"]
    assert [r["id"] for r in restored["rooms"]] == [room_id]
    assert restored["physical_rooms"][0]["id"] == physical[0]["id"]
    assert restored["meta"]["layout_rev"] == 2

    divided = client.post(
        f"/api/v1/jobs/{job_id}/dividers",
        json={
            "start_x": 2.5,
            "start_y": -1.0,
            "end_x": 2.5,
            "end_y": 5.0,
            "expected_layout_rev": 2,
        },
    )
    assert divided.status_code == 200, divided.text
    split_layout = client.get(f"/api/v1/jobs/{job_id}/layout").json()["data"]
    assert len(split_layout["physical_rooms"]) == len(physical)
    assert len(split_layout["rooms"]) == 2
    assert split_layout["virtual_dividers"]
    assert split_layout["meta"]["layout_rev"] == 3

    accepted = client.post(f"/api/v1/jobs/{job_id}/accept")
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["data"]["action"] == "accept"

    settings = get_settings()
    with psycopg.connect(settings.database_url) as conn:
        lake = compile_lake(
            conn=conn,
            object_store=ObjectStore(settings.object_store_dir),
            output_dir=tmp_path / "lake",
        )
    assert lake.face_cls
    assert lake.link_pred
    assert (tmp_path / "lake" / "face_cls.parquet").is_file()
    assert (tmp_path / "lake" / "link_pred.parquet").is_file()
    job = client.get(f"/api/v1/jobs/{job_id}")
    assert job.json()["data"]["layout_rev"] >= 4
