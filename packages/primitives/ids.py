"""Deterministic primitive IDs. Never use object identity or list position."""

from __future__ import annotations

from packages.common.geometry import geometry_hash_payload
from packages.common.hashing import stable_hash
from packages.ingest.models import Primitive


def assign_primitive_id(source_hash: str, primitive: Primitive, precision: int) -> str:
    payload = geometry_hash_payload(primitive.kind.value, primitive.geometry, precision)
    extra = {}
    if primitive.kind.value == "arc":
        extra = {
            "center": primitive.attributes.get("center"),
            "radius": primitive.attributes.get("radius"),
            "start_angle": primitive.attributes.get("start_angle"),
            "end_angle": primitive.attributes.get("end_angle"),
        }
    return stable_hash(
        kind=primitive.kind.value,
        rounded_coordinates=payload,
        relevant_attributes={
            "source_hash": source_hash,
            "source_handle": primitive.source_handle or "",
            "source_entity_type": primitive.source_entity_type,
            "layer": primitive.layer,
            "color": primitive.color if primitive.color is not None else "",
            "linetype": primitive.linetype or "",
            "block_name": primitive.block_name or "",
            **{k: v for k, v in extra.items() if v is not None},
        },
    )
