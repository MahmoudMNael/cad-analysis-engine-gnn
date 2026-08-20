from __future__ import annotations

from dataclasses import dataclass

from packages.contracts.enums import EntityClass
from packages.ingest.models import Primitive
from packages.models_nn.common import EntityScore


@dataclass(frozen=True)
class Measurement:
    id: str
    content: str | None
    layer: str
    source_entity_type: str


def collect_measurements(
    primitives: tuple[Primitive, ...],
    scores: dict[str, EntityScore],
) -> tuple[Measurement, ...]:
    out: list[Measurement] = []
    for prim in primitives:
        score = scores.get(prim.id)
        if prim.attributes.get("measurement") or (score and score.predicted_class == EntityClass.DIM.value):
            out.append(
                Measurement(
                    id=prim.id,
                    content=str(prim.attributes.get("content")) if prim.attributes.get("content") else None,
                    layer=prim.layer,
                    source_entity_type=prim.source_entity_type,
                )
            )
    return tuple(out)
