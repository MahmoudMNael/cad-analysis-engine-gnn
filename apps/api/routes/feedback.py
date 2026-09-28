"""Feedback event, divider, accept, and face-listing routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from apps.api.dependencies import event_store, job_repo, snapshot_store
from packages.contracts.dto.feedback import (
    DividerRequest,
    FeedbackEventListDto,
    FeedbackEventListResponse,
    FeedbackEventRequest,
    FeedbackEventResponse,
)
from packages.contracts.dto.layout import FaceListDto, FaceListResponse
from packages.contracts.enums import FeedbackAction, JobStatus
from packages.store.db.repositories.jobs import JobRepository
from packages.store.events import EventStore
from packages.store.feedback import load_faces, submit_feedback, undo_dividers
from packages.store.snapshots import SnapshotStore
from packages.common.errors import (
    JobFailedError,
    JobNotFoundError,
    JobNotReadyError,
)

router = APIRouter()


@router.post("/jobs/{job_id}/events", response_model=FeedbackEventResponse)
def create_event(
    job_id: str,
    body: FeedbackEventRequest,
    jobs: JobRepository = Depends(job_repo),
    events: EventStore = Depends(event_store),
    snaps: SnapshotStore = Depends(snapshot_store),
) -> FeedbackEventResponse:
    dto = submit_feedback(job_id, body, jobs=jobs, events=events, snapshots=snaps)
    return FeedbackEventResponse(success=True, data=dto, message="Event persisted")


@router.post("/jobs/{job_id}/dividers", response_model=FeedbackEventResponse)
def create_divider(
    job_id: str,
    body: DividerRequest,
    jobs: JobRepository = Depends(job_repo),
    events: EventStore = Depends(event_store),
    snaps: SnapshotStore = Depends(snapshot_store),
) -> FeedbackEventResponse:
    request = FeedbackEventRequest(
        action=FeedbackAction.SPLIT,
        target_ids=[],
        payload={
            "start_x": body.start_x,
            "start_y": body.start_y,
            "end_x": body.end_x,
            "end_y": body.end_y,
        },
        expected_layout_rev=body.expected_layout_rev,
        actor="user",
    )
    dto = submit_feedback(job_id, request, jobs=jobs, events=events, snapshots=snaps)
    return FeedbackEventResponse(success=True, data=dto, message="Divider persisted")


@router.delete("/jobs/{job_id}/dividers", response_model=FeedbackEventListResponse)
def delete_dividers(
    job_id: str,
    jobs: JobRepository = Depends(job_repo),
    events: EventStore = Depends(event_store),
    snaps: SnapshotStore = Depends(snapshot_store),
) -> FeedbackEventListResponse:
    written = undo_dividers(job_id, jobs=jobs, events=events, snapshots=snaps)
    layout_rev = jobs.layout_rev(job_id)
    return FeedbackEventListResponse(
        success=True,
        data=FeedbackEventListDto(events=written, layout_rev=layout_rev),
        message="Dividers removed" if written else "No dividers to remove",
    )


@router.post("/jobs/{job_id}/accept", response_model=FeedbackEventResponse)
def accept_job(
    job_id: str,
    jobs: JobRepository = Depends(job_repo),
    events: EventStore = Depends(event_store),
    snaps: SnapshotStore = Depends(snapshot_store),
) -> FeedbackEventResponse:
    request = FeedbackEventRequest(action=FeedbackAction.ACCEPT, target_ids=[], payload={}, actor="user")
    dto = submit_feedback(job_id, request, jobs=jobs, events=events, snapshots=snaps)
    return FeedbackEventResponse(success=True, data=dto, message="Accept recorded")


@router.get("/jobs/{job_id}/faces", response_model=FaceListResponse)
def list_faces(
    job_id: str,
    jobs: JobRepository = Depends(job_repo),
    snaps: SnapshotStore = Depends(snapshot_store),
) -> FaceListResponse:
    job = jobs.get(job_id)
    if job is None:
        raise JobNotFoundError(job_id)
    if job.status == JobStatus.FAILED.value:
        raise JobFailedError(job_id, job.error_detail or job.error_code or "failed")
    if job.status != JobStatus.SUCCEEDED.value:
        raise JobNotReadyError(job_id, job.status)
    faces = load_faces(job_id, snaps)
    return FaceListResponse(success=True, data=FaceListDto(faces=faces), message="")
