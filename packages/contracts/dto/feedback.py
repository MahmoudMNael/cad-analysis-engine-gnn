from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from packages.contracts.dto.common import ApiModel, RequestModel, require_non_empty_id
from packages.contracts.enums import FeedbackAction


class FeedbackEventRequest(RequestModel):
    action: FeedbackAction
    target_ids: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    expected_layout_rev: int | None = None
    actor: str = "user"

    @field_validator("target_ids")
    @classmethod
    def _targets(cls, value: list[str]) -> list[str]:
        return [require_non_empty_id(item) for item in value]


class DividerRequest(RequestModel):
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    expected_layout_rev: int | None = None


class FeedbackEventDto(ApiModel):
    event_id: str
    job_id: str
    sequence: int
    ts: str
    actor: str
    action: FeedbackAction
    target_ids: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    layout_rev: int
    retracted_by_event_id: str | None = None

    @field_validator("event_id", "job_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return require_non_empty_id(value)


class FeedbackEventResponse(ApiModel):
    success: bool = True
    data: FeedbackEventDto
    message: str = ""
