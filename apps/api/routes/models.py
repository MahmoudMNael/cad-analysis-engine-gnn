from __future__ import annotations

from fastapi import APIRouter, Depends

from apps.api.dependencies import model_repo
from packages.contracts.dto.models import ModelInfoDto, ModelListDto, ModelListResponse
from packages.contracts.enums import ModelRegistryStatus
from packages.eval.metrics import flatten_metrics
from packages.store.db.repositories.models import ModelRegistryRepository
from packages.store.models_registry import seed_heuristic_registry

router = APIRouter()


@router.get("/models", response_model=ModelListResponse)
def list_models(repo: ModelRegistryRepository = Depends(model_repo)) -> ModelListResponse:
    seed_heuristic_registry(repo)
    rows = repo.list_all()
    models = [
        ModelInfoDto(
            model_name=row["model_name"],
            version=row["version"],
            status=ModelRegistryStatus(row["status"]),
            checkpoint_uri=row.get("checkpoint_uri"),
            feature_schema_version=row["feature_schema_version"],
            metrics=flatten_metrics(row.get("metrics_json") or {}),
            trained_on_engine_sha=row.get("trained_on_engine_sha"),
            created_at=row["created_at"].isoformat() if row.get("created_at") else None,
            promoted_at=row["promoted_at"].isoformat() if row.get("promoted_at") else None,
        )
        for row in rows
    ]
    return ModelListResponse(success=True, data=ModelListDto(models=models), message="")
