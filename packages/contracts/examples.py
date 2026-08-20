"""OpenAPI examples for public DTOs."""

from packages.contracts.enums import JobStatus, UnitSource, WarningCode, WarningSeverity

JOB_CREATED_EXAMPLE = {
    "success": True,
    "data": {"job_id": "job_example", "status": JobStatus.QUEUED},
    "message": "Job queued",
}

LAYOUT_EXAMPLE = {
    "unit": "m",
    "meta": {
        "job_id": "job_example",
        "source_hash": "a" * 64,
        "engine_sha": "b" * 64,
        "layout_schema": "layout.v1",
        "pipeline_version": "1.0.0",
        "unit": "m",
        "unit_source": UnitSource.CAD_METADATA,
        "scale_factor": 0.001,
        "model_versions": {
            "entity": "heuristic:entity_v1",
            "gap": "heuristic:gap_v1",
            "face": "heuristic:face_v1",
        },
        "model_decision_sources": {
            "entity": "heuristic",
            "gap": "heuristic",
            "face": "heuristic",
        },
        "feature_schema_versions": {
            "entity": "entity_features_v1",
            "gap": "gap_features_v1",
            "face": "face_features_v1",
        },
        "configuration_hash": "c" * 64,
        "created_at": "2026-01-01T00:00:00+00:00",
        "layout_rev": 0,
    },
    "rooms": [],
    "physical_rooms": [],
    "walls": [],
    "doors": [],
    "windows": [],
    "objects": [],
    "unassigned_objects": [],
    "texts": [],
    "lights": [],
    "virtual_dividers": [],
    "warnings": [
        {
            "code": WarningCode.NO_ROOMS_FOUND,
            "severity": WarningSeverity.WARNING,
            "message": "No physical rooms were assembled",
            "target_ids": [],
        }
    ],
}
