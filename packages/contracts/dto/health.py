from __future__ import annotations

from typing import Literal

from packages.contracts.dto.common import ApiModel


class ComponentHealthDto(ApiModel):
    status: Literal["ok", "error"]
    detail: str | None = None


class HealthDataDto(ApiModel):
    status: Literal["ok", "degraded", "error"]
    postgres: ComponentHealthDto
    redis: ComponentHealthDto
    object_store: ComponentHealthDto


class HealthResponse(ApiModel):
    success: bool = True
    data: HealthDataDto
    message: str = ""
