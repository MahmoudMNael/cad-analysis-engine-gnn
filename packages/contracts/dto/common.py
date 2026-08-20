from __future__ import annotations

import math
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from packages.contracts.enums import ErrorCode, WarningCode, WarningSeverity

T = TypeVar("T")


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PointDto(ApiModel):
    x: float
    y: float

    @field_validator("x", "y")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("coordinates must be finite")
        return float(value)


class BBoxDto(ApiModel):
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @model_validator(mode="after")
    def _ordered(self) -> BBoxDto:
        if not all(math.isfinite(v) for v in (self.min_x, self.min_y, self.max_x, self.max_y)):
            raise ValueError("bbox coordinates must be finite")
        if self.min_x > self.max_x or self.min_y > self.max_y:
            raise ValueError("bbox min must be <= max")
        return self


class WarningDto(ApiModel):
    code: WarningCode
    severity: WarningSeverity
    message: str
    target_ids: list[str] = Field(default_factory=list)


class ErrorBodyDto(ApiModel):
    code: ErrorCode
    detail: str
    fields: dict[str, str] = Field(default_factory=dict)


class ErrorEnvelopeDto(ApiModel):
    success: bool = False
    error: ErrorBodyDto


class SuccessEnvelopeDto(ApiModel, Generic[T]):
    success: bool = True
    data: T
    message: str = ""


def require_non_empty_id(value: str) -> str:
    if not value or not str(value).strip():
        raise ValueError("IDs must be non-empty")
    return str(value)
