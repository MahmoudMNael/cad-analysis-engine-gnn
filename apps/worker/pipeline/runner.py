"""Exact P1 pipeline order. No stage may silently skip another."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packages.assemble.face_features import (
    build_face_feature_index,
    extract_face_features,
    feature_schema_version,
)
from packages.assemble.layout import build_layout_dto
from packages.assemble.measurements import collect_measurements
from packages.assemble.objects import assemble_objects
from packages.assemble.openings import attach_openings
from packages.assemble.polygonize import polygonize_graph
from packages.assemble.rooms import assemble_physical_rooms
from packages.assemble.texts import assemble_texts
from packages.assemble.warnings import compile_warnings
from packages.common.configuration import Settings
from packages.contracts.dto.layout import LayoutDto
from packages.contracts.enums import EntityClass, WarningCode, WarningSeverity
from packages.contracts.dto.common import WarningDto
from packages.graphkit.candidates import generate_gap_candidates
from packages.graphkit.graph_builder import build_plan_graph, close_graph
from packages.graphkit.models import PlanGraph
from packages.graphkit.portals import detect_portals
from packages.ingest.dxf_loader import build_document
from packages.ingest.models import Document, Primitive
from packages.models_nn.common import EntityScore, FaceScore, GapScore
from packages.models_nn.features.entity import ENTITY_FEATURE_SCHEMA_VERSION
from packages.models_nn.features.face import build_face_adjacency
from packages.models_nn.features.gap import GAP_FEATURE_SCHEMA_VERSION
from packages.models_nn.runtime import EntityRuntime, FaceRuntime, GapRuntime
from packages.primitives.features import extract_entity_features
from packages.primitives.models import primitive_set_from_document
from packages.primitives.spatial_index import PrimitiveIndexes
from packages.views.reducer import reduce_view
from apps.worker.pipeline.context import PipelineContext


@dataclass
class PipelineResult:
    document: Document
    layout: LayoutDto
    plan_graph_payload: dict[str, Any]
    document_payload: dict[str, Any]
    faces_rows: dict[str, list[Any]]
    meta: dict[str, Any]


def _geom_payload(prim: Primitive) -> dict[str, Any]:
    geom = prim.geometry
    return {
        "wkt": geom.wkt,
        "type": geom.geom_type,
    }


def serialize_document(document: Document) -> dict[str, Any]:
    return {
        "document_id": document.document_id,
        "source_hash": document.source_hash,
        "format": document.format.value,
        "unit": document.unit,
        "unit_source": document.unit_source.value,
        "bbox": document.bbox.as_tuple(),
        "metadata": dict(document.metadata),
        "primitives": [
            {
                "id": p.id,
                "kind": p.kind.value,
                "layer": p.layer,
                "color": p.color,
                "linetype": p.linetype,
                "source_entity_type": p.source_entity_type,
                "source_handle": p.source_handle,
                "block_name": p.block_name,
                "attributes": dict(p.attributes),
                "bbox": p.bbox.as_tuple(),
                "length": p.length,
                "geometry": _geom_payload(p),
            }
            for p in document.primitives
        ],
    }


def serialize_plan_graph(
    graph: PlanGraph,
    entity_scores: dict[str, EntityScore],
    gap_scores: dict[str, GapScore],
) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "id": n.id,
                "point": n.point,
                "incident_edge_ids": list(n.incident_edge_ids),
                "source_primitive_ids": list(n.source_primitive_ids),
            }
            for n in graph.nodes
        ],
        "edges": [
            {
                "id": e.id,
                "node_a": e.node_a,
                "node_b": e.node_b,
                "wkt": e.geometry.wkt,
                "primitive_ids": list(e.primitive_ids),
                "class_prior": e.class_prior,
                "predicted_class": e.predicted_class,
                "confidence": e.confidence,
            }
            for e in graph.edges
        ],
        "portals": [
            {
                "id": p.id,
                "kind": p.kind.value,
                "wkt": p.geometry.wkt,
                "source_edge_ids": list(p.source_edge_ids),
                "confidence": p.confidence,
                "source_primitive_ids": list(p.source_primitive_ids),
            }
            for p in graph.portals
        ],
        "metadata": dict(graph.metadata),
        "entity_scores": {
            k: {
                "probabilities": v.probabilities,
                "predicted_class": v.predicted_class,
                "confidence": v.confidence,
                "entropy": v.entropy,
                "model_version": v.model_version,
                "decision_source": v.decision_source.value,
                "explanation": v.explanation,
            }
            for k, v in entity_scores.items()
        },
        "gap_scores": {
            k: {
                "probability": v.probability,
                "decision": v.decision.value,
                "confidence": v.confidence,
                "model_version": v.model_version,
                "decision_source": v.decision_source.value,
                "explanation": v.explanation,
            }
            for k, v in gap_scores.items()
        },
    }


def _classify_primitive(
    prim: Primitive,
    by_id: dict[str, Primitive],
    indexes: PrimitiveIndexes,
    entity_runtime: EntityRuntime,
    settings: Settings,
) -> tuple[str, EntityScore]:
    feats = extract_entity_features(
        prim,
        by_id,
        indexes,
        include_neighbors=entity_runtime.requires_neighbor_features,
    )
    return prim.id, entity_runtime.predict(prim, feats, settings)


def classify_entities(
    primitives: tuple[Primitive, ...],
    by_id: dict[str, Primitive],
    indexes: PrimitiveIndexes,
    entity_runtime: EntityRuntime,
    settings: Settings,
) -> dict[str, EntityScore]:
    workers = max(1, int(settings.entity_classify_workers or 1))
    if not entity_runtime.requires_neighbor_features or workers <= 1 or len(primitives) < 2:
        return dict(
            _classify_primitive(prim, by_id, indexes, entity_runtime, settings) for prim in primitives
        )

    def _one(prim: Primitive) -> tuple[str, EntityScore]:
        return _classify_primitive(prim, by_id, indexes, entity_runtime, settings)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(_one, primitives))


def run_pipeline(
    *,
    source_path: Path,
    context: PipelineContext,
    settings: Settings,
    explicit_unit: str | None = None,
) -> PipelineResult:
    entity_ref = context.model_versions.get("entity_model_version", "heuristic:entity_v1")
    gap_ref = context.model_versions.get("gap_model_version", "heuristic:gap_v1")
    face_ref = context.model_versions.get("face_model_version", "heuristic:face_v1")
    entity_runtime = EntityRuntime(entity_ref)
    gap_runtime = GapRuntime(gap_ref)
    face_runtime = FaceRuntime(face_ref, checkpoint_uri=context.model_checkpoint_uris.get("face"))

    with context.track_stage("ingest", model_ref="heuristic:ingest_v1", count_in=1) as stage:
        document, ingest_warnings = build_document(
            source_path, context.source_hash, settings, explicit_unit=explicit_unit
        )
        context.add_warnings(ingest_warnings)
        stage.count_out = len(document.primitives)

    with context.track_stage(
        "primitive_indexing",
        model_ref="heuristic:index_v1",
        count_in=len(document.primitives),
    ) as stage:
        primitive_set = primitive_set_from_document(document)
        by_id = primitive_set.by_id()
        stage.count_out = len(document.primitives)

    with context.track_stage(
        "entity_classification",
        model_ref=entity_ref,
        count_in=len(document.primitives),
    ) as stage:
        entity_scores = classify_entities(
            document.primitives, by_id, primitive_set.indexes, entity_runtime, settings
        )
        stage.count_out = len(entity_scores)

    with context.track_stage(
        "graph_construction",
        model_ref="heuristic:graph_v1",
        count_in=len(document.primitives),
    ) as stage:
        graph = build_plan_graph(document.primitives, entity_scores, settings)
        stage.count_out = len(graph.edges)

    with context.track_stage(
        "gap_candidate_generation",
        model_ref="heuristic:gap_candidates_v1",
        count_in=len(graph.nodes),
    ) as stage:
        candidates = generate_gap_candidates(graph, settings)
        cand_by_id = {c.pair_id: c for c in candidates}
        stage.count_out = len(candidates)

    with context.track_stage("gap_classification", model_ref=gap_ref, count_in=len(candidates)) as stage:
        gap_scores: dict[str, GapScore] = {}
        for cand in candidates:
            gap_scores[cand.pair_id] = gap_runtime.predict(
                cand.pair_id,
                cand.distance_m,
                collinear=cand.collinear,
                crosses=cand.crosses,
                already_connected=cand.already_connected,
                settings=settings,
                opening_like=cand.opening_like,
            )
        stage.count_out = len(gap_scores)

    with context.track_stage("graph_closure", model_ref="heuristic:closure_v1", count_in=len(gap_scores)) as stage:
        gap_geoms = {c.pair_id: c.geometry for c in candidates}
        portals = detect_portals(graph, document.primitives, entity_scores, gap_scores, gap_geoms, settings)
        graph = close_graph(graph, gap_scores, cand_by_id, settings, extra_portals=portals)
        stage.count_out = len(graph.edges)

    with context.track_stage("polygonization", model_ref="heuristic:polygonize_v1", count_in=len(graph.edges)) as stage:
        faces, poly_notes = polygonize_graph(graph, settings)
        if poly_notes:
            context.add_warnings(
                [
                    WarningDto(
                        code=WarningCode.CAD_PARSE_WARNING,
                        severity=WarningSeverity.INFO,
                        message=f"polygonize note: {note}",
                        target_ids=[],
                    )
                    for note in poly_notes
                ]
            )
        stage.count_out = len(faces.faces)

    texts_prims = tuple(p for p in document.primitives if p.kind.value == "text")
    wall_prims = tuple(
        p
        for p in document.primitives
        if entity_scores.get(p.id) and entity_scores[p.id].predicted_class == EntityClass.WALL.value
    )

    with context.track_stage(
        "face_feature_extraction",
        model_ref="heuristic:face_features_v1",
        count_in=len(faces.faces),
    ) as stage:
        face_feature_map: dict[str, dict[str, float]] = {}
        feature_index = build_face_feature_index(faces, graph, texts_prims, wall_prims)
        for face in faces.faces:
            face_feature_map[face.id] = extract_face_features(
                face, faces, graph, graph.portals, texts_prims, wall_prims, index=feature_index
            )
        stage.count_out = len(face_feature_map)

    with context.track_stage("face_classification", model_ref=face_ref, count_in=len(faces.faces)) as stage:
        face_ids = [face.id for face in faces.faces]
        edge_index, edge_attr = build_face_adjacency(
            [(face.id, face.geometry, face.area, face.perimeter) for face in faces.faces],
            [portal.geometry for portal in graph.portals],
        )
        face_scores = face_runtime.predict_graph(
            face_ids,
            face_feature_map,
            edge_index,
            edge_attr,
            settings,
        )
        stage.count_out = len(face_scores)

    with context.track_stage("room_assembly", model_ref="heuristic:rooms_v1", count_in=len(faces.faces)) as stage:
        rooms = assemble_physical_rooms(faces, face_scores)
        stage.count_out = len(rooms)

    with context.track_stage("openings", model_ref="heuristic:openings_v1", count_in=len(graph.portals)) as stage:
        rooms, doors, windows = attach_openings(rooms, graph.portals)
        stage.count_out = len(doors) + len(windows)

    with context.track_stage("objects", model_ref="heuristic:objects_v1", count_in=len(document.primitives)) as stage:
        objects, unassigned, rooms = assemble_objects(document.primitives, entity_scores, rooms, settings)
        stage.count_out = len(objects) + len(unassigned)

    with context.track_stage("texts", model_ref="heuristic:texts_v1", count_in=len(texts_prims)) as stage:
        texts, rooms = assemble_texts(document.primitives, entity_scores, rooms)
        stage.count_out = len(texts)

    with context.track_stage(
        "measurements",
        model_ref="heuristic:measurements_v1",
        count_in=len(document.primitives),
    ) as stage:
        measurements = collect_measurements(document.primitives, entity_scores)
        stage.count_out = len(measurements)

    with context.track_stage("warnings", model_ref="heuristic:warnings_v1", count_in=0) as stage:
        warnings = compile_warnings(
            document=document,
            ingest_warnings=list(context.warnings),
            entity_scores=entity_scores,
            gap_scores=gap_scores,
            face_scores=face_scores,
            faces=faces,
            rooms=rooms,
            unassigned=unassigned,
            entity_threshold=settings.entity_threshold,
            face_threshold=settings.face_threshold,
        )
        context.warnings = warnings
        stage.count_out = len(warnings)

    feature_versions = {
        "entity": ENTITY_FEATURE_SCHEMA_VERSION,
        "gap": GAP_FEATURE_SCHEMA_VERSION,
        "face": feature_schema_version(),
    }
    decision_sources = {
        "entity": "heuristic" if entity_ref.startswith("heuristic:") else "model",
        "gap": "heuristic" if gap_ref.startswith("heuristic:") else "model",
        "face": face_runtime.decision_source_name,
    }
    config_hash = context.configuration if isinstance(context.configuration, str) else ""

    with context.track_stage("layout_assembly", model_ref="heuristic:layout_v1", count_in=len(rooms)) as stage:
        physical_layout = build_layout_dto(
            job_id=context.job_id,
            document=document,
            graph=graph,
            rooms=rooms,
            doors=doors,
            windows=windows,
            objects=objects,
            unassigned=unassigned,
            texts=texts,
            warnings=warnings,
            engine_sha=context.engine_sha,
            pipeline_version=settings.pipeline_version,
            layout_schema=settings.layout_schema,
            model_versions=context.model_versions,
            model_decision_sources=decision_sources,
            feature_schema_versions=feature_versions,
            configuration_hash=config_hash,
        )
        stage.count_out = len(physical_layout.physical_rooms)

    with context.track_stage(
        "initial_derived_view",
        model_ref="heuristic:view_v1",
        count_in=len(physical_layout.physical_rooms),
    ) as stage:
        layout = reduce_view(physical_layout, ())
        LayoutDto.model_validate(layout.model_dump())
        stage.count_out = len(layout.rooms)

    with context.track_stage(
        "serialization",
        model_ref="heuristic:serialize_v1",
        count_in=len(faces.faces),
    ) as stage:
        faces_rows: dict[str, list[Any]] = {
            "face_id": [],
            "wkt": [],
            "area": [],
            "perimeter": [],
            "predicted_class": [],
            "confidence": [],
            "entropy": [],
            "decision_source": [],
            "explanation": [],
            "accepted_as_room": [],
            "repaired": [],
            "discarded_reason": [],
            "valid": [],
            "probabilities": [],
            "features": [],
        }
        room_ids = {r.id for r in rooms}
        for face in faces.faces:
            score = face_scores[face.id]
            faces_rows["face_id"].append(face.id)
            faces_rows["wkt"].append(face.geometry.wkt)
            faces_rows["area"].append(face.area)
            faces_rows["perimeter"].append(face.perimeter)
            faces_rows["predicted_class"].append(score.predicted_class)
            faces_rows["confidence"].append(score.confidence)
            faces_rows["entropy"].append(score.entropy)
            faces_rows["decision_source"].append(score.decision_source.value)
            faces_rows["explanation"].append(score.explanation)
            faces_rows["accepted_as_room"].append(face.id in room_ids)
            faces_rows["repaired"].append(face.repaired)
            faces_rows["discarded_reason"].append(face.discarded_reason)
            faces_rows["valid"].append(face.valid)
            faces_rows["probabilities"].append(json.dumps(score.probabilities, sort_keys=True))
            faces_rows["features"].append(json.dumps(face_feature_map[face.id], sort_keys=True))

        document_payload = serialize_document(document)
        document_payload["measurements"] = [
            {"id": m.id, "layer": m.layer, "source_entity_type": m.source_entity_type, "content": m.content}
            for m in measurements
        ]
        plan_graph_payload = serialize_plan_graph(graph, entity_scores, gap_scores)
        stage.count_out = len(faces_rows["face_id"])
    meta = {
        "job_id": context.job_id,
        "source_hash": context.source_hash,
        "engine_sha": context.engine_sha,
        "layout_schema": settings.layout_schema,
        "pipeline_version": settings.pipeline_version,
        "feature_schema_versions": feature_versions,
        "model_versions": context.model_versions,
        "model_decision_sources": decision_sources,
        "unit": document.unit,
        "unit_source": document.unit_source.value,
        "created_at": layout.meta.created_at,
        "stage_timings": context.timings.as_dict() | {k: v["duration_ms"] for k, v in context.block_meta.items()},
        "warnings": [w.model_dump(mode="json") for w in warnings],
        "configuration_hash": config_hash,
        "success": True,
        "thresholds": settings.job_config_dict(),
        "block_meta": context.block_meta,
    }
    return PipelineResult(
        document=document,
        layout=layout,
        plan_graph_payload=plan_graph_payload,
        document_payload=document_payload,
        faces_rows=faces_rows,
        meta=meta,
    )
