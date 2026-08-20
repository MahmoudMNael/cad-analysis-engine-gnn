from __future__ import annotations

import math
from typing import Any

from pydantic import Field, field_validator, model_validator

from packages.contracts.dto.common import (
    ApiModel,
    BBoxDto,
    PointDto,
    WarningDto,
    require_non_empty_id,
)
from packages.contracts.enums import PortalKind, UnitSource


def _finite_nonneg(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return float(value)


def _confidence(value: float) -> float:
    if not math.isfinite(value) or value < 0 or value > 1:
        raise ValueError("confidence must be in [0, 1]")
    return float(value)


class DimensionsDto(ApiModel):
    width_m: float | None = None
    length_m: float | None = None

    @field_validator("width_m", "length_m")
    @classmethod
    def _pos(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return _finite_nonneg(value, "dimension")


class RoomDto(ApiModel):
    id: str
    boundary: list[PointDto]
    area_m2: float
    area_net_m2: float
    area_gross_m2: float
    center: PointDto
    perimeter_m: float
    bbox: BBoxDto
    confidence: float
    name: str | None = None
    kind: str
    selectable: bool = True
    neighbor_ids: list[str] = Field(default_factory=list)
    door_ids: list[str] = Field(default_factory=list)
    window_ids: list[str] = Field(default_factory=list)
    object_ids: list[str] = Field(default_factory=list)
    physical_room_id: str | None = None
    parent_room_id: str | None = None
    dimensions: DimensionsDto | None = None

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        return require_non_empty_id(value)

    @field_validator("boundary")
    @classmethod
    def _boundary(cls, value: list[PointDto]) -> list[PointDto]:
        if len(value) < 3:
            raise ValueError("polygon vertices >= 3")
        return value

    @field_validator("area_m2", "area_net_m2", "area_gross_m2", "perimeter_m")
    @classmethod
    def _areas(cls, value: float, info: Any) -> float:
        return _finite_nonneg(value, info.field_name)

    @field_validator("confidence")
    @classmethod
    def _conf(cls, value: float) -> float:
        return _confidence(value)


class WallDto(ApiModel):
    id: str
    start: PointDto
    end: PointDto
    primitive_ids: list[str] = Field(default_factory=list)
    predicted_class: str = "wall"
    confidence: float = 1.0

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        return require_non_empty_id(value)

    @field_validator("confidence")
    @classmethod
    def _conf(cls, value: float) -> float:
        return _confidence(value)


class OpeningDto(ApiModel):
    id: str
    kind: PortalKind
    start: PointDto
    end: PointDto
    source_edge_ids: list[str] = Field(default_factory=list)
    confidence: float = 1.0

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        return require_non_empty_id(value)

    @field_validator("confidence")
    @classmethod
    def _conf(cls, value: float) -> float:
        return _confidence(value)


class ObjectDto(ApiModel):
    id: str
    kind: str = "furniture"
    center: PointDto
    bbox: BBoxDto
    layer: str | None = None
    block_name: str | None = None
    room_id: str | None = None
    confidence: float = 1.0

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        return require_non_empty_id(value)

    @field_validator("confidence")
    @classmethod
    def _conf(cls, value: float) -> float:
        return _confidence(value)


class TextDto(ApiModel):
    id: str
    content: str
    position: PointDto
    bbox: BBoxDto
    rotation: float = 0.0
    layer: str | None = None
    confidence: float = 1.0
    room_id: str | None = None

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        return require_non_empty_id(value)

    @field_validator("confidence")
    @classmethod
    def _conf(cls, value: float) -> float:
        return _confidence(value)

    @field_validator("rotation")
    @classmethod
    def _rot(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("rotation must be finite")
        return float(value)


class LightDto(ApiModel):
    id: str
    position: PointDto
    confidence: float = 1.0

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        return require_non_empty_id(value)

    @field_validator("confidence")
    @classmethod
    def _conf(cls, value: float) -> float:
        return _confidence(value)


class VirtualDividerDto(ApiModel):
    id: str
    start: PointDto
    end: PointDto

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        return require_non_empty_id(value)


class LayoutMetaDto(ApiModel):
    job_id: str
    source_hash: str
    engine_sha: str
    layout_schema: str
    pipeline_version: str
    unit: str
    unit_source: UnitSource
    scale_factor: float
    model_versions: dict[str, str] = Field(default_factory=dict)
    model_decision_sources: dict[str, str] = Field(default_factory=dict)
    feature_schema_versions: dict[str, str] = Field(default_factory=dict)
    configuration_hash: str
    created_at: str
    layout_rev: int = 0

    @field_validator("job_id", "source_hash", "engine_sha")
    @classmethod
    def _ids(cls, value: str) -> str:
        return require_non_empty_id(value)

    @model_validator(mode="after")
    def _unit_is_metres(self) -> LayoutMetaDto:
        if self.unit != "m":
            raise ValueError("layout metadata unit must be m")
        if not math.isfinite(self.scale_factor) or self.scale_factor <= 0:
            raise ValueError("scale_factor must be a positive finite number")
        return self


class LayoutDto(ApiModel):
    unit: str
    meta: LayoutMetaDto
    rooms: list[RoomDto] = Field(default_factory=list)
    physical_rooms: list[RoomDto] = Field(default_factory=list)
    walls: list[WallDto] = Field(default_factory=list)
    doors: list[OpeningDto] = Field(default_factory=list)
    windows: list[OpeningDto] = Field(default_factory=list)
    objects: list[ObjectDto] = Field(default_factory=list)
    unassigned_objects: list[ObjectDto] = Field(default_factory=list)
    texts: list[TextDto] = Field(default_factory=list)
    lights: list[LightDto] = Field(default_factory=list)
    virtual_dividers: list[VirtualDividerDto] = Field(default_factory=list)
    warnings: list[WarningDto] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unit(self) -> LayoutDto:
        if self.unit != "m":
            raise ValueError("layout unit must be m")
        return self


class LayoutResponse(ApiModel):
    success: bool = True
    data: LayoutDto
    message: str = ""


class FaceDto(ApiModel):
    id: str
    boundary: list[PointDto]
    area_m2: float
    predicted_class: str
    confidence: float
    decision_source: str
    accepted_as_room: bool
    adjacent_unbounded: bool = False

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        return require_non_empty_id(value)

    @field_validator("boundary")
    @classmethod
    def _boundary(cls, value: list[PointDto]) -> list[PointDto]:
        if len(value) < 3:
            raise ValueError("polygon vertices >= 3")
        return value

    @field_validator("area_m2")
    @classmethod
    def _area(cls, value: float) -> float:
        return _finite_nonneg(value, "area_m2")

    @field_validator("confidence")
    @classmethod
    def _conf(cls, value: float) -> float:
        return _confidence(value)


class FaceListDto(ApiModel):
    faces: list[FaceDto] = Field(default_factory=list)


class FaceListResponse(ApiModel):
    success: bool = True
    data: FaceListDto
    message: str = ""
