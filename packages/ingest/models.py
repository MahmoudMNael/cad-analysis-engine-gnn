from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from shapely.geometry.base import BaseGeometry

from packages.common.geometry import bbox_tuple, length_of
from packages.contracts.enums import CadFormat, PrimitiveKind, UnitSource


@dataclass(frozen=True)
class BBox:
    minx: float
    miny: float
    maxx: float
    maxy: float

    def extent(self) -> float:
        return max(self.maxx - self.minx, self.maxy - self.miny)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.minx, self.miny, self.maxx, self.maxy)


def bbox_from_geom(geom: BaseGeometry) -> BBox:
    minx, miny, maxx, maxy = bbox_tuple(geom)
    return BBox(minx, miny, maxx, maxy)


@dataclass(frozen=True)
class Primitive:
    id: str
    kind: PrimitiveKind
    geometry: BaseGeometry
    layer: str
    color: int | None
    linetype: str | None
    source_entity_type: str
    source_handle: str | None
    block_name: str | None
    attributes: Mapping[str, Any]
    bbox: BBox
    length: float


@dataclass(frozen=True)
class Document:
    document_id: str
    source_hash: str
    format: CadFormat
    unit: str
    unit_source: UnitSource
    bbox: BBox
    primitives: tuple[Primitive, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def scale_factor(self) -> float:
        return float(self.metadata.get("scale_factor", 1.0))


def primitive_from_parts(
    *,
    primitive_id: str,
    kind: PrimitiveKind,
    geometry: BaseGeometry,
    layer: str,
    color: int | None,
    linetype: str | None,
    source_entity_type: str,
    source_handle: str | None,
    block_name: str | None,
    attributes: Mapping[str, Any],
) -> Primitive:
    return Primitive(
        id=primitive_id,
        kind=kind,
        geometry=geometry,
        layer=layer,
        color=color,
        linetype=linetype,
        source_entity_type=source_entity_type,
        source_handle=source_handle,
        block_name=block_name,
        attributes=dict(attributes),
        bbox=bbox_from_geom(geometry),
        length=length_of(geometry),
    )
