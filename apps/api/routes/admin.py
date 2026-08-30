"""Local admin APIs for GNN-C dataset export, kernel source, and candidate import.

Training itself is not triggered here. POST /admin/train remains unavailable (P6).
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import PlainTextResponse

from apps.api.dependencies import db_conn, get_engine_sha, get_object_store
from apps.trainer.export import export_face_dataset
from apps.trainer.train import register_imported_face_checkpoint
from packages.common.configuration import Settings, get_settings
from packages.common.errors import AppError, InvalidRequestError
from packages.contracts.dto.models import (
    FaceCheckpointMetricsDto,
    FaceDatasetDto,
    FaceDatasetResponse,
    ModelImportResponse,
    ModelInfoDto,
    TrainRequest,
)
from packages.contracts.enums import ErrorCode, ModelRegistryStatus
from packages.eval.metrics import flatten_metrics
from packages.models_nn.face_train_kernel import KERNEL_PATH
from packages.store.object_store import ObjectStore

router = APIRouter()


@router.get("/admin/datasets/face/kernel.py", response_class=PlainTextResponse)
def face_train_kernel_source() -> str:
    return KERNEL_PATH.read_text(encoding="utf-8")


@router.get("/admin/datasets/face", response_model=FaceDatasetResponse)
def export_face_graphs(
    conn=Depends(db_conn),
    store: ObjectStore = Depends(get_object_store),
    settings: Settings = Depends(get_settings),
    engine_sha: str = Depends(get_engine_sha),
) -> FaceDatasetResponse:
    payload = export_face_dataset(
        conn=conn,
        object_store=store,
        settings=settings,
        engine_sha=engine_sha,
    )
    return FaceDatasetResponse(
        success=True,
        data=FaceDatasetDto.model_validate(payload),
        message="",
    )


@router.post("/admin/models", response_model=ModelImportResponse)
async def import_face_model(
    checkpoint: UploadFile = File(...),
    metrics: UploadFile = File(...),
    conn=Depends(db_conn),
    store: ObjectStore = Depends(get_object_store),
    engine_sha: str = Depends(get_engine_sha),
) -> ModelImportResponse:
    checkpoint_bytes = await checkpoint.read()
    raw_metrics = await metrics.read()
    try:
        metrics_payload = json.loads(raw_metrics.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidRequestError("metrics must be a JSON object") from exc
    if not isinstance(metrics_payload, dict):
        raise InvalidRequestError("metrics must be a JSON object")
    parsed = FaceCheckpointMetricsDto.model_validate(metrics_payload)
    row = register_imported_face_checkpoint(
        conn=conn,
        object_store=store,
        checkpoint_bytes=checkpoint_bytes,
        metrics=metrics_payload,
        engine_sha=parsed.engine_sha or engine_sha,
    )
    return ModelImportResponse(success=True, data=_model_info(row), message="")


@router.post("/admin/train")
def train(body: TrainRequest) -> None:
    raise AppError(ErrorCode.MODEL_NOT_AVAILABLE, "training is not available in P1")


def _model_info(row: dict) -> ModelInfoDto:
    created = row.get("created_at")
    promoted = row.get("promoted_at")
    return ModelInfoDto(
        model_name=row["model_name"],
        version=row["version"],
        status=ModelRegistryStatus(row["status"]),
        checkpoint_uri=row.get("checkpoint_uri"),
        feature_schema_version=row["feature_schema_version"],
        metrics=flatten_metrics(row.get("metrics_json") or {}),
        trained_on_engine_sha=row.get("trained_on_engine_sha"),
        created_at=created.isoformat() if created is not None else None,
        promoted_at=promoted.isoformat() if promoted is not None else None,
    )
