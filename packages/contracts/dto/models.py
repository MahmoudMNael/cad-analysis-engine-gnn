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


class FaceGraphDto(ApiModel):
    job_id: str
    building_id: str
    face_ids: list[str]
    node_features: list[list[float]]
    edge_index: list[list[int]]
    edge_attr: list[list[float]]
    labels: list[int]
    weights: list[float]


class DatasetEligibilityDto(ApiModel):
    ok: bool
    labeled_jobs: int
    n_classes: int
    n_labeled_faces: int
    reasons: list[str] = Field(default_factory=list)


class FaceTrainHyperparamsDto(ApiModel):
    face_train_min_labeled_jobs: int
    face_train_min_classes: int
    face_train_ratio: float
    face_val_ratio: float
    face_train_seed: int
    face_gnn_hidden_dim: int
    face_gnn_layers: int
    face_gnn_dropout: float
    face_gnn_epochs: int
    face_gnn_lr: float
    cavity_area_m2: float
    shaft_area_m2: float
    outdoor_area_m2: float
    face_threshold: float


class FaceDatasetDto(ApiModel):
    feature_schema_version: str
    edge_feature_schema_version: str
    node_feature_keys: list[str]
    edge_feature_keys: list[str]
    classes: list[str]
    engine_sha: str
    hyperparams: FaceTrainHyperparamsDto
    eligibility: DatasetEligibilityDto
    graphs: list[FaceGraphDto] = Field(default_factory=list)


class FaceDatasetResponse(ApiModel):
    success: bool = True
    data: FaceDatasetDto
    message: str = ""


class FaceCheckpointMetricsDto(RequestModel):
    feature_schema_version: str
    engine_sha: str | None = None
    version: str | None = None
    macro_f1: float | None = None
    per_class_f1: dict[str, float] | None = None
    confusion_matrix: list[list[int]] | None = None
    n: float | None = None
    room_count_error: float | None = None
    false_room_rate: float | None = None
    delete_rate_proxy: float | None = None
    room_iou_at_0_5: float | None = None
    heuristic_macro_f1: float | None = None
    heuristic_delete_rate_proxy: float | None = None
    heuristic_f1_indoor: float | None = None
    held_out_buildings: float | None = None
    train_macro_f1: float | None = None
    best_val_macro_f1: float | None = None
    held_out_passed: float | bool | None = None
    n_train_graphs: float | None = None
    n_val_graphs: float | None = None
    n_test_graphs: float | None = None
    f1_indoor: float | None = None
    f1_corridor: float | None = None
    f1_shaft: float | None = None
    f1_outdoor: float | None = None
    f1_cavity: float | None = None

    @field_validator("feature_schema_version")
    @classmethod
    def _schema(cls, value: str) -> str:
        if value != "face_features_v1":
            raise ValueError("must be face_features_v1")
        return value


class ModelImportResponse(ApiModel):
    success: bool = True
    data: ModelInfoDto
    message: str = ""
