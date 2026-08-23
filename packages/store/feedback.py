"""Submit validated feedback events and compute the derived layout."""

from __future__ import annotations

from packages.common.errors import (
    JobFailedError,
    JobNotFoundError,
    JobNotReadyError,
    LayoutNotFoundError,
)
from packages.common.logging import get_logger
from packages.contracts.dto.feedback import FeedbackEventDto, FeedbackEventRequest
from packages.contracts.dto.layout import FaceDto, LayoutDto
from packages.contracts.enums import FeedbackAction, JobStatus
from packages.store.db.repositories.jobs import JobRepository
from packages.store.events import EventAppend, EventStore
from packages.store.snapshots import SnapshotStore
from packages.views.reducer import reduce_view
from packages.views.validate import divider_events_to_undo, prepare_feedback

log = get_logger("store.feedback")


def _require_succeeded(job_id: str, jobs: JobRepository):
    job = jobs.get(job_id)
    if job is None:
        raise JobNotFoundError(job_id)
    if job.status == JobStatus.FAILED.value:
        raise JobFailedError(job_id, job.error_detail or job.error_code or "failed")
    if job.status != JobStatus.SUCCEEDED.value:
        raise JobNotReadyError(job_id, job.status)
    return job


def derived_layout(
    job_id: str,
    *,
    events: EventStore,
    snapshots: SnapshotStore,
) -> LayoutDto:
    try:
        physical = snapshots.load_layout(job_id)
    except Exception as exc:
        raise LayoutNotFoundError(job_id) from exc
    return reduce_view(physical, events.list_dtos(job_id))


def load_faces(job_id: str, snapshots: SnapshotStore) -> list[FaceDto]:
    return snapshots.load_faces(job_id)


def submit_feedback(
    job_id: str,
    request: FeedbackEventRequest,
    *,
    jobs: JobRepository,
    events: EventStore,
    snapshots: SnapshotStore,
) -> FeedbackEventDto:
    created = submit_feedback_batch(
        job_id,
        [request],
        jobs=jobs,
        events=events,
        snapshots=snapshots,
        expected_layout_rev=request.expected_layout_rev,
    )
    return created[0]


def submit_feedback_batch(
    job_id: str,
    requests: list[FeedbackEventRequest],
    *,
    jobs: JobRepository,
    events: EventStore,
    snapshots: SnapshotStore,
    expected_layout_rev: int | None = None,
) -> list[FeedbackEventDto]:
    _require_succeeded(job_id, jobs)
    try:
        physical = snapshots.load_layout(job_id)
    except Exception as exc:
        raise LayoutNotFoundError(job_id) from exc
    faces = snapshots.load_faces(job_id)
    try:
        plan_graph = snapshots.load_plan_graph(job_id)
    except Exception:
        plan_graph = {}
    working_events = events.list_dtos(job_id)
    working_layout = reduce_view(physical, working_events)
    records: list[EventAppend] = []
    for request in requests:
        prepared = prepare_feedback(
            action=request.action,
            target_ids=list(request.target_ids),
            payload=dict(request.payload),
            actor=request.actor,
            layout=working_layout,
            events=working_events,
            faces=faces,
            plan_graph=plan_graph,
        )
        records.append(
            EventAppend(
                action=prepared.action.value,
                target_ids=prepared.target_ids,
                payload=prepared.payload,
                actor=prepared.actor,
                retract_event_id=prepared.retract_event_id,
                clear_retraction_id=prepared.clear_retraction_id,
            )
        )
        pending = FeedbackEventDto(
            event_id=f"pending-{len(working_events) + 1}",
            job_id=job_id,
            sequence=len(working_events) + 1,
            ts="1970-01-01T00:00:00+00:00",
            actor=prepared.actor,
            action=prepared.action,
            target_ids=prepared.target_ids,
            payload=prepared.payload,
            layout_rev=working_layout.meta.layout_rev + 1,
            retracted_by_event_id=None,
        )
        updated: list[FeedbackEventDto] = []
        for event in working_events:
            retracted_by = event.retracted_by_event_id
            if prepared.retract_event_id and event.event_id == prepared.retract_event_id:
                retracted_by = pending.event_id
            if prepared.clear_retraction_id and event.event_id == prepared.clear_retraction_id:
                retracted_by = None
            if retracted_by != event.retracted_by_event_id:
                event = event.model_copy(update={"retracted_by_event_id": retracted_by})
            updated.append(event)
        working_events = [*updated, pending]
        working_layout = reduce_view(physical, working_events)
    written = events.append_batch(job_id, records, expected_layout_rev=expected_layout_rev)
    log.info(
        "feedback_appended",
        extra={
            "job_id": job_id,
            "stage": "feedback",
            "block": ",".join(record.action for record in records),
            "count_in": len(requests),
            "count_out": len(written),
        },
    )
    return written


def undo_dividers(
    job_id: str,
    *,
    jobs: JobRepository,
    events: EventStore,
    snapshots: SnapshotStore,
) -> list[FeedbackEventDto]:
    _require_succeeded(job_id, jobs)
    to_undo = divider_events_to_undo(events.list_dtos(job_id))
    if not to_undo:
        return []
    requests = [
        FeedbackEventRequest(action=FeedbackAction.UNDO, target_ids=[event.event_id], actor="user")
        for event in to_undo
    ]
    return submit_feedback_batch(job_id, requests, jobs=jobs, events=events, snapshots=snapshots)
