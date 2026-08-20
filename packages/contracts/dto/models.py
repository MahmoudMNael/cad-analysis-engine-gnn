from __future__ import annotations

from pydantic import Field, field_validator

from packages.contracts.dto.common import ApiModel, RequestModel, require_non_empty_id
from packages.contracts.enums import ModelRegistryStatus


class ModelInfoDto(ApiModel):
    model_name: str
    version: str
    status: ModelRegistryStatus
    checkpoint_uri: str | None = None
    feature_schema_version: str
    metrics: dict[str, float] = Field(default_factory=dict)
    trained_on_engine_sha: str | None = None
    created_at: str | None = None
    promoted_at: str | None = None

    @field_validator("model_name", "version", "feature_schema_version")
    @classmethod
    def _ids(cls, value: str) -> str:
        return require_non_empty_id(value)


class ModelListDto(ApiModel):
    models: list[ModelInfoDto] = Field(default_factory=list)


class ModelListResponse(ApiModel):
    success: bool = True
    data: ModelListDto
    message: str = ""


class TrainRequest(RequestModel):
    model_name: str = "face"
    force: bool = False


class TrainAcceptedDto(ApiModel):
    training_run_id: str
    status: str

    @field_validator("training_run_id")
    @classmethod
    def _id(cls, value: str) -> str:
        return require_non_empty_id(value)


class TrainAcceptedResponse(ApiModel):
    success: bool = True
    data: TrainAcceptedDto
    message: str = ""
