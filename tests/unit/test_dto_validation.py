from math import isclose

from pydantic import ValidationError

from packages.contracts.dto.common import BBoxDto, PointDto, WarningDto
from packages.contracts.dto.layout import LayoutDto, LayoutMetaDto, RoomDto
from packages.contracts.enums import UnitSource, WarningCode, WarningSeverity


def _meta() -> LayoutMetaDto:
    return LayoutMetaDto(
        job_id="job1",
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


def test_room_rejects_short_polygon():
    try:
        RoomDto(
            id="r1",
            boundary=[PointDto(x=0, y=0), PointDto(x=1, y=0)],
            area_m2=1,
            area_net_m2=1,
            area_gross_m2=1,
            center=PointDto(x=0.5, y=0.5),
            perimeter_m=4,
            bbox=BBoxDto(min_x=0, min_y=0, max_x=1, max_y=1),
            confidence=0.9,
            kind="indoor",
        )
        raise AssertionError("expected validation error")
    except ValidationError:
        pass


def test_confidence_bounds():
    try:
        RoomDto(
            id="r1",
            boundary=[PointDto(x=0, y=0), PointDto(x=1, y=0), PointDto(x=1, y=1)],
            area_m2=1,
            area_net_m2=1,
            area_gross_m2=1,
            center=PointDto(x=0.5, y=0.5),
            perimeter_m=4,
            bbox=BBoxDto(min_x=0, min_y=0, max_x=1, max_y=1),
            confidence=1.2,
            kind="indoor",
        )
        raise AssertionError("expected validation error")
    except ValidationError:
        pass


def test_bbox_min_le_max():
    try:
        BBoxDto(min_x=2, min_y=0, max_x=1, max_y=1)
        raise AssertionError("expected validation error")
    except ValidationError:
        pass


def test_layout_requires_metres():
    try:
        LayoutDto(unit="mm", meta=_meta())
        raise AssertionError("expected validation error")
    except ValidationError:
        pass


def test_warning_dto_roundtrip():
    warning = WarningDto(
        code=WarningCode.NO_ROOMS_FOUND,
        severity=WarningSeverity.WARNING,
        message="none",
        target_ids=[],
    )
    assert warning.code is WarningCode.NO_ROOMS_FOUND
    assert isclose(1.0, 1.0)
