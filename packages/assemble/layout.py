from __future__ import annotations

from datetime import datetime, timezone

from packages.assemble.objects import PlacedObject
from packages.assemble.rooms import PhysicalRoom
from packages.assemble.texts import LayoutText
from packages.assemble.validation import assert_layout_invariants
from packages.common.geometry import aspect_ratio
from packages.contracts.dto.common import BBoxDto, PointDto, WarningDto
from packages.contracts.dto.layout import (
    DimensionsDto,
    LayoutDto,
    LayoutMetaDto,
    LightDto,
    ObjectDto,
    OpeningDto,
    RoomDto,
    TextDto,
    WallDto,
)
from packages.contracts.enums import UnitSource
from packages.graphkit.models import PlanGraph, Portal
from packages.ingest.models import Document


def _bbox(bounds: tuple[float, float, float, float]) -> BBoxDto:
    return BBoxDto(min_x=bounds[0], min_y=bounds[1], max_x=bounds[2], max_y=bounds[3])


def _ring(geom) -> list[PointDto]:
    coords = list(geom.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [PointDto(x=float(x), y=float(y)) for x, y, *_ in coords]


def _room_dto(room: PhysicalRoom, selectable: bool = True) -> RoomDto:
    minx, miny, maxx, maxy = room.bbox
    return RoomDto(
        id=room.id,
        boundary=_ring(room.geometry),
        area_m2=room.area_m2,
        area_net_m2=room.area_m2,
        area_gross_m2=room.area_m2,
        center=PointDto(x=room.center[0], y=room.center[1]),
        perimeter_m=room.perimeter_m,
        bbox=_bbox(room.bbox),
        confidence=room.confidence,
        name=room.name,
        kind=room.kind,
        selectable=selectable,
        neighbor_ids=list(room.neighbor_ids),
        door_ids=list(room.door_ids),
        window_ids=list(room.window_ids),
        object_ids=list(room.object_ids),
        physical_room_id=room.id,
        parent_room_id=None,
        dimensions=DimensionsDto(
            width_m=max(maxx - minx, 0.0),
            length_m=max(maxy - miny, 0.0),
        ),
    )


def _opening_dto(portal: Portal) -> OpeningDto:
    coords = list(portal.geometry.coords) if portal.geometry.geom_type == "LineString" else [
        (portal.geometry.bounds[0], portal.geometry.bounds[1]),
        (portal.geometry.bounds[2], portal.geometry.bounds[3]),
    ]
    start, end = coords[0], coords[-1]
    return OpeningDto(
        id=portal.id,
        kind=portal.kind,
        start=PointDto(x=float(start[0]), y=float(start[1])),
        end=PointDto(x=float(end[0]), y=float(end[1])),
        source_edge_ids=list(portal.source_edge_ids),
        confidence=portal.confidence,
    )


def _object_dto(obj: PlacedObject) -> ObjectDto:
    return ObjectDto(
        id=obj.id,
        kind=obj.kind,
        center=PointDto(x=obj.center[0], y=obj.center[1]),
        bbox=_bbox(obj.bbox),
        layer=obj.layer,
        block_name=obj.block_name,
        room_id=obj.room_id,
        confidence=obj.confidence,
    )


def build_layout_dto(
    *,
    job_id: str,
    document: Document,
    graph: PlanGraph,
    rooms: tuple[PhysicalRoom, ...],
    doors: tuple[Portal, ...],
    windows: tuple[Portal, ...],
    objects: tuple[PlacedObject, ...],
    unassigned: tuple[PlacedObject, ...],
    texts: tuple[LayoutText, ...],
    warnings: list[WarningDto],
    engine_sha: str,
    pipeline_version: str,
    layout_schema: str,
    model_versions: dict[str, str],
    model_decision_sources: dict[str, str],
    feature_schema_versions: dict[str, str],
    configuration_hash: str,
    created_at: datetime | None = None,
) -> LayoutDto:
    walls = []
    for edge in graph.edges:
        if edge.predicted_class == "portal":
            continue
        coords = list(edge.geometry.coords)
        walls.append(
            WallDto(
                id=edge.id,
                start=PointDto(x=float(coords[0][0]), y=float(coords[0][1])),
                end=PointDto(x=float(coords[-1][0]), y=float(coords[-1][1])),
                primitive_ids=list(edge.primitive_ids),
                predicted_class=edge.predicted_class or "wall",
                confidence=edge.confidence,
            )
        )
    physical = [_room_dto(room, selectable=True) for room in rooms]
    stamp = (created_at or datetime.now(timezone.utc)).isoformat()
    layout = LayoutDto(
        unit="m",
        meta=LayoutMetaDto(
            job_id=job_id,
            source_hash=document.source_hash,
            engine_sha=engine_sha,
            layout_schema=layout_schema,
            pipeline_version=pipeline_version,
            unit="m",
            unit_source=document.unit_source,
            scale_factor=document.scale_factor,
            model_versions=model_versions,
            model_decision_sources=model_decision_sources,
            feature_schema_versions=feature_schema_versions,
            configuration_hash=configuration_hash,
            created_at=stamp,
            layout_rev=0,
        ),
        rooms=list(physical),
        physical_rooms=list(physical),
        walls=walls,
        doors=[_opening_dto(p) for p in doors],
        windows=[_opening_dto(p) for p in windows],
        objects=[_object_dto(o) for o in objects],
        unassigned_objects=[_object_dto(o) for o in unassigned],
        texts=[
            TextDto(
                id=t.id,
                content=t.content,
                position=PointDto(x=t.position[0], y=t.position[1]),
                bbox=_bbox(t.bbox),
                rotation=t.rotation,
                layer=t.layer,
                confidence=t.confidence,
                room_id=t.room_id,
            )
            for t in texts
        ],
        lights=[],
        virtual_dividers=[],
        warnings=warnings,
    )
    assert_layout_invariants(layout)
    return layout
