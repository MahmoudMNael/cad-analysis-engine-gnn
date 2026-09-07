"""Stable pipeline fingerprints for output-identity goldens."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from apps.worker.pipeline.runner import PipelineResult

VOLATILE_LAYOUT_META = ("engine_sha", "created_at", "job_id", "source_hash")


def _dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_dumps(payload).encode("utf-8")).hexdigest()


def stable_layout(layout_dict: dict[str, Any]) -> dict[str, Any]:
    meta = dict(layout_dict.get("meta") or {})
    for key in VOLATILE_LAYOUT_META:
        meta.pop(key, None)
    out = copy.deepcopy(layout_dict)
    out["meta"] = meta
    for wall in out.get("walls") or []:
        wall["primitive_ids"] = []
    for key in ("rooms", "physical_rooms"):
        for room in out.get(key) or []:
            room["object_ids"] = []
    for key in ("objects", "unassigned_objects", "texts"):
        for item in out.get(key) or []:
            item.pop("id", None)
    return out


def _stable_document(payload: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(payload)
    out.pop("source_hash", None)
    out.pop("document_id", None)
    for prim in out.get("primitives") or []:
        prim.pop("id", None)
    for meas in out.get("measurements") or []:
        meas.pop("id", None)
    return out


def _stable_plan_graph(payload: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(payload)
    for edge in out.get("edges") or []:
        edge["primitive_ids"] = []
    for node in out.get("nodes") or []:
        node["source_primitive_ids"] = []
    for portal in out.get("portals") or []:
        portal["source_primitive_ids"] = []
    scores = out.get("entity_scores") or {}
    out["entity_scores"] = sorted(scores.values(), key=_dumps)
    return out


def pipeline_fingerprint(result: PipelineResult) -> dict[str, Any]:
    layout = stable_layout(result.layout.model_dump(mode="json"))
    return {
        "layout": layout,
        "faces": {
            "face_id": list(result.faces_rows["face_id"]),
            "predicted_class": list(result.faces_rows["predicted_class"]),
            "accepted_as_room": list(result.faces_rows["accepted_as_room"]),
            "features": list(result.faces_rows["features"]),
        },
        "primitive_kinds": sorted(p.kind.value for p in result.document.primitives),
        "document_sha256": _sha256(_stable_document(result.document_payload)),
        "plan_graph_sha256": _sha256(_stable_plan_graph(result.plan_graph_payload)),
    }


def write_pipeline_goldens(out_dir: Path, tmp_dir: Path) -> None:
    from tests.helpers.dxf_builder import write_simple_dxf
    from tests.helpers.pipeline import run_dxf
    from tests.helpers.synthetic_cases import PIPELINE_CASES

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, builder in PIPELINE_CASES:
        path = write_simple_dxf(tmp_dir / f"{name}.dxf", builder)
        result = run_dxf(path, job_id=f"golden-{name}")
        payload = pipeline_fingerprint(result)
        (out_dir / f"{name}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
