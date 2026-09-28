from __future__ import annotations

from packages.contracts.dto.common import BBoxDto, PointDto
from packages.contracts.dto.layout import LayoutDto, LayoutMetaDto, RoomDto
from packages.contracts.enums import UnitSource


def meta(job_id: str = "job1") -> LayoutMetaDto:
    return LayoutMetaDto(
        job_id=job_id,
        source_hash="a" * 64,
        engine_sha="b" * 64,
        layout_schema="layout.v1",
        pipeline_version="1.0.0",
        unit="m",
        unit_source=UnitSource.CAD_METADATA,
        scale_factor=0.001,
        configuration_hash="c" * 64,
        created_at="2026-01-01T00:00:00+00:00",
    )


def rect_room(
    room_id: str,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    *,
    kind: str = "indoor",
    name: str | None = None,
    physical_room_id: str | None = None,
) -> RoomDto:
    return RoomDto(
        id=room_id,
        boundary=[
            PointDto(x=minx, y=miny),
            PointDto(x=maxx, y=miny),
            PointDto(x=maxx, y=maxy),
            PointDto(x=minx, y=maxy),
        ],
        area_m2=(maxx - minx) * (maxy - miny),
        area_net_m2=(maxx - minx) * (maxy - miny),
        area_gross_m2=(maxx - minx) * (maxy - miny),
        center=PointDto(x=(minx + maxx) / 2.0, y=(miny + maxy) / 2.0),
        perimeter_m=2.0 * ((maxx - minx) + (maxy - miny)),
        bbox=BBoxDto(min_x=minx, min_y=miny, max_x=maxx, max_y=maxy),
        confidence=0.9,
        name=name,
        kind=kind,
        selectable=True,
        physical_room_id=physical_room_id or room_id,
    )


def layout_from_rooms(*rooms: RoomDto, job_id: str = "job1") -> LayoutDto:
    copied = list(rooms)
    return LayoutDto(
        unit="m",
        meta=meta(job_id),
        rooms=list(copied),
        physical_rooms=list(copied),
    )
