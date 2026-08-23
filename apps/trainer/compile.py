"""Compile snapshot + explicit event history into the feedback lake."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from packages.contracts.dto.feedback import FeedbackEventDto
from packages.contracts.dto.layout import LayoutDto
from packages.contracts.enums import FeedbackAction, LabelSource
from packages.eval.datasets import (
    CompiledLake,
    TrainingRow,
    label_source_for,
    make_sample_id,
    weight_for,
)
from packages.models_nn.features.entity import ENTITY_FEATURE_SCHEMA_VERSION
from packages.models_nn.features.face import FACE_FEATURE_SCHEMA_VERSION
from packages.models_nn.features.gap import GAP_FEATURE_SCHEMA_VERSION, extract_gap_features
from packages.store.db.repositories.jobs import JobRepository
from packages.store.events import EventStore
from packages.store.object_store import ObjectStore
from packages.store.snapshots import SnapshotStore, _parse_json_map
from packages.views.reducer import reduce_view


def compile_job_artifacts(
    *,
    job_id: str,
    building_id: str,
    engine_sha: str,
    consent_training: bool,
    physical_layout: LayoutDto,
    faces_columns: dict[str, list[Any]],
    plan_graph: dict[str, Any],
    document: dict[str, Any],
    events: list[FeedbackEventDto],
    require_consent: bool = True,
) -> CompiledLake:
    if require_consent and not consent_training:
        return CompiledLake(face_cls=[], edge_cls=[], link_pred=[])
    face_features = _face_feature_map(faces_columns)
    lake = CompiledLake(face_cls=[], edge_cls=[], link_pred=[])
    physical_ids = {room.id for room in physical_layout.physical_rooms}
    unretracted = [event for event in events if event.retracted_by_event_id is None]
    for event in unretracted:
        source = label_source_for(event.actor, event.action)
        weight = weight_for(source)
        action = event.action
        if action == FeedbackAction.UNDO:
            continue
        if action == FeedbackAction.RENAME_ROOM:
            continue
        if action == FeedbackAction.DELETE_ROOM:
            for target in event.target_ids:
                face_id = _physical_face_id(physical_layout, target)
                lake.face_cls.append(
                    _face_row(
                        job_id=job_id,
                        building_id=building_id,
                        engine_sha=engine_sha,
                        target_id=face_id,
                        label=str(event.payload.get("label") or "not_indoor"),
                        source=source,
                        weight=weight,
                        sequence=event.sequence,
                        action=action.value,
                        features=face_features.get(face_id, {}),
                    )
                )
        elif action == FeedbackAction.RESTORE_FACE:
            for target in event.target_ids:
                lake.face_cls.append(
                    _face_row(
                        job_id=job_id,
                        building_id=building_id,
                        engine_sha=engine_sha,
                        target_id=target,
                        label=str(event.payload.get("label") or "indoor"),
                        source=source,
                        weight=weight,
                        sequence=event.sequence,
                        action=action.value,
                        features=face_features.get(target, {}),
                    )
                )
        elif action == FeedbackAction.MERGE_ROOMS:
            target_id = "|".join(sorted(event.target_ids))
            lake.edge_cls.append(
                _row(
                    dataset="edge_cls",
                    job_id=job_id,
                    building_id=building_id,
                    engine_sha=engine_sha,
                    target_id=target_id,
                    schema=FACE_FEATURE_SCHEMA_VERSION,
                    model_input_version=FACE_FEATURE_SCHEMA_VERSION,
                    features={"member_count": float(len(event.target_ids))},
                    label="not_room_boundary",
                    source=source,
                    weight=weight,
                    sequence=event.sequence,
                    action=action.value,
                )
            )
        elif action == FeedbackAction.SPLIT:
            for target in event.target_ids:
                face_id = _physical_face_id(physical_layout, target)
                lake.face_cls.append(
                    _face_row(
                        job_id=job_id,
                        building_id=building_id,
                        engine_sha=engine_sha,
                        target_id=face_id,
                        label="indoor",
                        source=source,
                        weight=weight,
                        sequence=event.sequence,
                        action=action.value,
                        features=face_features.get(face_id, {}),
                    )
                )
            pair = (
                (float(event.payload.get("start_x", 0.0)), float(event.payload.get("start_y", 0.0))),
                (float(event.payload.get("end_x", 0.0)), float(event.payload.get("end_y", 0.0))),
            )
            distance = ((pair[0][0] - pair[1][0]) ** 2 + (pair[0][1] - pair[1][1]) ** 2) ** 0.5
            lake.link_pred.append(
                _row(
                    dataset="link_pred",
                    job_id=job_id,
                    building_id=building_id,
                    engine_sha=engine_sha,
                    target_id="|".join(sorted(event.target_ids)) or "divider",
                    schema=GAP_FEATURE_SCHEMA_VERSION,
                    model_input_version=GAP_FEATURE_SCHEMA_VERSION,
                    features=extract_gap_features(
                        distance_m=distance,
                        d_max=max(distance, 1.0),
                        collinear=True,
                        already_connected=False,
                        crosses=False,
                    ),
                    label="close",
                    source=source,
                    weight=weight,
                    sequence=event.sequence,
                    action=action.value,
                )
            )
        elif action in {FeedbackAction.DRAW_WALL, FeedbackAction.CONNECT_NODES}:
            distance = _payload_distance(event.payload)
            target_id = "|".join(event.target_ids) if event.target_ids else _payload_pair_key(event.payload)
            lake.link_pred.append(
                _row(
                    dataset="link_pred",
                    job_id=job_id,
                    building_id=building_id,
                    engine_sha=engine_sha,
                    target_id=target_id,
                    schema=GAP_FEATURE_SCHEMA_VERSION,
                    model_input_version=GAP_FEATURE_SCHEMA_VERSION,
                    features=extract_gap_features(
                        distance_m=distance,
                        d_max=max(distance, 1.0),
                        collinear=True,
                        already_connected=False,
                        crosses=False,
                    ),
                    label="close",
                    source=source,
                    weight=weight,
                    sequence=event.sequence,
                    action=action.value,
                )
            )
        elif action == FeedbackAction.MARK_ENTITY:
            label = str(event.payload.get("class") or event.payload.get("label") or "unknown")
            for target in event.target_ids:
                lake.edge_cls.append(
                    _row(
                        dataset="edge_cls",
                        job_id=job_id,
                        building_id=building_id,
                        engine_sha=engine_sha,
                        target_id=target,
                        schema=ENTITY_FEATURE_SCHEMA_VERSION,
                        model_input_version=ENTITY_FEATURE_SCHEMA_VERSION,
                        features=_primitive_features(document, target),
                        label=label,
                        source=source,
                        weight=weight,
                        sequence=event.sequence,
                        action=action.value,
                    )
                )
        elif action == FeedbackAction.ACCEPT:
            derived = reduce_view(physical_layout, events)
            for room in derived.rooms:
                face_id = room.physical_room_id or room.id
                if face_id not in physical_ids and face_id not in face_features:
                    continue
                lake.face_cls.append(
                    _face_row(
                        job_id=job_id,
                        building_id=building_id,
                        engine_sha=engine_sha,
                        target_id=face_id,
                        label="indoor",
                        source=LabelSource.WEAK,
                        weight=weight_for(LabelSource.WEAK),
                        sequence=event.sequence,
                        action=action.value,
                        features=face_features.get(face_id, {}),
                    )
                )
    return lake


def compile_lake(
    *,
    conn: Any,
    object_store: ObjectStore,
    output_dir: Path | None = None,
    require_consent: bool = True,
) -> CompiledLake:
    jobs = JobRepository(conn)
    events = EventStore(conn)
    snapshots = SnapshotStore(object_store, jobs)
    merged = CompiledLake(face_cls=[], edge_cls=[], link_pred=[])
    for job in jobs.list_succeeded():
        if require_consent and not job.consent_training:
            continue
        document = snapshots.load_document(job.id)
        meta = snapshots.load_meta(job.id)
        building_id = str(
            (document.get("metadata") or {}).get("building_id")
            or meta.get("building_id")
            or job.id
        )
        part = compile_job_artifacts(
            job_id=job.id,
            building_id=building_id,
            engine_sha=job.engine_sha or str(meta.get("engine_sha") or ""),
            consent_training=job.consent_training,
            physical_layout=snapshots.load_layout(job.id),
            faces_columns=snapshots.load_faces_columns(job.id),
            plan_graph=snapshots.load_plan_graph(job.id),
            document=document,
            events=events.list_dtos(job.id),
            require_consent=require_consent,
        )
        merged.face_cls.extend(part.face_cls)
        merged.edge_cls.extend(part.edge_cls)
        merged.link_pred.extend(part.link_pred)
    directory = Path(output_dir) if output_dir is not None else Path(object_store.root) / "lake"
    merged.write_parquet(directory)
    return merged


def _face_feature_map(columns: dict[str, list[Any]]) -> dict[str, dict[str, Any]]:
    ids = columns.get("face_id") or []
    raw_features = columns.get("features") or [None] * len(ids)
    out: dict[str, dict[str, Any]] = {}
    for i, face_id in enumerate(ids):
        out[str(face_id)] = _parse_json_map(raw_features[i] if i < len(raw_features) else None)
    return out


def _physical_face_id(layout: LayoutDto, target_id: str) -> str:
    for room in (*layout.rooms, *layout.physical_rooms):
        if room.id == target_id:
            return room.physical_room_id or room.id
        if room.physical_room_id == target_id:
            return room.physical_room_id
    return target_id


def _payload_distance(payload: dict[str, Any]) -> float:
    try:
        x0, y0 = float(payload["start_x"]), float(payload["start_y"])
        x1, y1 = float(payload["end_x"]), float(payload["end_y"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    return float(((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5)


def _payload_pair_key(payload: dict[str, Any]) -> str:
    return f"{payload.get('start_x')},{payload.get('start_y')}|{payload.get('end_x')},{payload.get('end_y')}"


def _primitive_features(document: dict[str, Any], primitive_id: str) -> dict[str, Any]:
    for prim in document.get("primitives") or []:
        if prim.get("id") == primitive_id:
            return {
                "length": float(prim.get("length") or 0.0),
                "color": float(prim.get("color") or 0.0),
            }
    return {}


def _face_row(**kwargs: Any) -> TrainingRow:
    return _row(
        dataset="face_cls",
        schema=FACE_FEATURE_SCHEMA_VERSION,
        model_input_version=FACE_FEATURE_SCHEMA_VERSION,
        **kwargs,
    )


def _row(
    *,
    dataset: str,
    job_id: str,
    building_id: str,
    engine_sha: str,
    target_id: str,
    schema: str,
    model_input_version: str,
    features: dict[str, Any],
    label: str,
    source: LabelSource,
    weight: float,
    sequence: int,
    action: str,
) -> TrainingRow:
    return TrainingRow(
        sample_id=make_sample_id(
            job_id=job_id,
            dataset=dataset,
            target_id=target_id,
            sequence=sequence,
            action=action,
        ),
        job_id=job_id,
        building_id=building_id,
        target_id=target_id,
        feature_schema_version=schema,
        features=features,
        label=label,
        label_source=source,
        weight=weight,
        engine_sha=engine_sha,
        model_input_version=model_input_version,
    )


def main() -> None:
    import psycopg

    from packages.common.configuration import get_settings

    settings = get_settings()
    store = ObjectStore(settings.object_store_dir)
    with psycopg.connect(settings.database_url) as conn:
        lake = compile_lake(conn=conn, object_store=store)
    print(
        json.dumps(
            {
                "face_cls": len(lake.face_cls),
                "edge_cls": len(lake.edge_cls),
                "link_pred": len(lake.link_pred),
            }
        )
    )


if __name__ == "__main__":
    main()
