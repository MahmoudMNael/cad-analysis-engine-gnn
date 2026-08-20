from __future__ import annotations

from dataclasses import dataclass

from packages.ingest.models import Document, Primitive
from packages.primitives.spatial_index import PrimitiveIndexes, build_primitive_indexes


@dataclass(frozen=True)
class PrimitiveSet:
    document: Document
    primitives: tuple[Primitive, ...]
    indexes: PrimitiveIndexes

    def by_id(self) -> dict[str, Primitive]:
        return {p.id: p for p in self.primitives}


def primitive_set_from_document(document: Document) -> PrimitiveSet:
    return PrimitiveSet(
        document=document,
        primitives=document.primitives,
        indexes=build_primitive_indexes(document.primitives),
    )
