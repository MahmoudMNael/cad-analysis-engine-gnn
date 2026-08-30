from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import ValidationError

from apps.api.app import create_app
from packages.contracts.dto.models import FaceCheckpointMetricsDto
from packages.models_nn.face_train_kernel import KERNEL_PATH


def test_kernel_source_is_standalone_and_contains_face_gnn():
    source = KERNEL_PATH.read_text(encoding="utf-8")
    assert "class FaceGNN" in source
    assert "import packages" not in source
    assert "from packages" not in source
    assert "packages." not in source


def test_kernel_endpoint_serves_repo_source():
    client = TestClient(create_app())
    response = client.get("/api/v1/admin/datasets/face/kernel.py")
    assert response.status_code == 200, response.text
    body = response.text
    assert "class FaceGNN" in body
    assert body == KERNEL_PATH.read_text(encoding="utf-8")
    assert "text/" in (response.headers.get("content-type") or "")


def test_notebook_exists():
    path = Path(__file__).resolve().parents[2] / "notebooks" / "gnn_c_train.ipynb"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "BASE_URL" in text
    assert "FORCE" in text
    assert "/api/v1/admin/datasets/face/kernel.py" in text
    assert "face_gnn.pt" in text
    assert "metrics.json" in text


def test_import_metrics_require_face_schema():
    FaceCheckpointMetricsDto(feature_schema_version="face_features_v1", macro_f1=0.8)
    try:
        FaceCheckpointMetricsDto(feature_schema_version="face_features_v0")
        raise AssertionError("expected validation error")
    except ValidationError:
        pass
    try:
        FaceCheckpointMetricsDto(feature_schema_version="face_features_v1", unexpected=True)
        raise AssertionError("expected extra field rejection")
    except ValidationError:
        pass
