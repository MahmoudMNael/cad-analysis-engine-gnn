from fastapi.openapi.utils import get_openapi

from apps.api.app import create_app


REQUIRED_PATHS = {
    "/api/v1/health",
    "/api/v1/jobs",
    "/api/v1/jobs/from-url",
    "/api/v1/jobs/{job_id}",
    "/api/v1/jobs/{job_id}/cancel",
    "/api/v1/jobs/{job_id}/layout",
    "/api/v1/jobs/{job_id}/events",
    "/api/v1/jobs/{job_id}/dividers",
    "/api/v1/jobs/{job_id}/accept",
    "/api/v1/jobs/{job_id}/faces",
    "/api/v1/models",
    "/api/v1/admin/train",
    "/api/v1/admin/datasets/face",
    "/api/v1/admin/datasets/face/kernel.py",
    "/api/v1/admin/models",
}


def test_openapi_contains_required_paths_and_is_stable(tmp_path):
    app = create_app()
    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    paths = set(schema["paths"].keys())
    missing = REQUIRED_PATHS - paths
    assert not missing, missing
    snapshot = tmp_path.parent.parent / "golden" / "openapi.json"
    # committed snapshot lives in tests/golden/openapi.json
    from pathlib import Path
    import json

    committed = Path(__file__).resolve().parents[1] / "golden" / "openapi.json"
    dumped = json.dumps(schema, indent=2, sort_keys=True)
    if not committed.exists():
        committed.parent.mkdir(parents=True, exist_ok=True)
        committed.write_text(dumped + "\n", encoding="utf-8")
    expected = committed.read_text(encoding="utf-8")
    assert dumped + "\n" == expected or dumped == expected.strip() or json.loads(expected) == schema
