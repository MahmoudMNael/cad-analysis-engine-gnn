"""Feedback routes are specified for OpenAPI completeness; behavior is P2."""

from __future__ import annotations

from fastapi import APIRouter

from packages.common.errors import InvalidRequestError
from packages.contracts.dto.feedback import DividerRequest, FeedbackEventRequest

router = APIRouter()


@router.post("/jobs/{job_id}/events")
def create_event(job_id: str, body: FeedbackEventRequest) -> None:
    raise InvalidRequestError("feedback events are implemented in P2")


@router.post("/jobs/{job_id}/dividers")
def create_divider(job_id: str, body: DividerRequest) -> None:
    raise InvalidRequestError("dividers are implemented in P2")


@router.delete("/jobs/{job_id}/dividers")
def delete_dividers(job_id: str) -> None:
    raise InvalidRequestError("dividers are implemented in P2")


@router.post("/jobs/{job_id}/accept")
def accept_job(job_id: str) -> None:
    raise InvalidRequestError("accept is implemented in P2")


@router.get("/jobs/{job_id}/faces")
def list_faces(job_id: str) -> None:
    raise InvalidRequestError("face listing is implemented in P2")
