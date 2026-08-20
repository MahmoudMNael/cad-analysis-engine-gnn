"""Unit detection, provenance, and conversion to metres."""

from __future__ import annotations

from dataclasses import dataclass

from ezdxf import units as ez_units
from ezdxf.enums import InsertUnits

from packages.common.configuration import Settings
from packages.common.errors import ConversionError, InvalidRequestError
from packages.contracts.enums import UnitSource

UNIT_ALIASES: dict[str, str] = {
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "in": "in",
    "inch": "in",
    "inches": "in",
    "ft": "ft",
    "foot": "ft",
    "feet": "ft",
}

UNIT_TO_INSERT: dict[str, InsertUnits] = {
    "m": InsertUnits.Meters,
    "mm": InsertUnits.Millimeters,
    "cm": InsertUnits.Centimeters,
    "in": InsertUnits.Inches,
    "ft": InsertUnits.Feet,
}

INSUNITS_TO_NAME: dict[int, str] = {
    int(InsertUnits.Inches): "in",
    int(InsertUnits.Feet): "ft",
    int(InsertUnits.Miles): "mi",
    int(InsertUnits.Millimeters): "mm",
    int(InsertUnits.Centimeters): "cm",
    int(InsertUnits.Meters): "m",
    int(InsertUnits.Kilometers): "km",
    int(InsertUnits.Yards): "yd",
    int(InsertUnits.Decimeters): "dm",
}


@dataclass(frozen=True)
class UnitResolution:
    unit: str
    unit_source: UnitSource
    scale_factor: float
    cad_insunits: int | None
    original_unit: str


def normalize_unit_name(raw: str) -> str:
    key = raw.strip().lower()
    if key not in UNIT_ALIASES:
        raise InvalidRequestError(f"unsupported unit '{raw}'", fields={"unit": raw})
    return UNIT_ALIASES[key]


def metres_factor(unit_name: str) -> float:
    insert = UNIT_TO_INSERT.get(unit_name)
    if insert is None:
        known = {
            "km": InsertUnits.Kilometers,
            "yd": InsertUnits.Yards,
            "dm": InsertUnits.Decimeters,
            "mi": InsertUnits.Miles,
        }
        insert = known.get(unit_name)
    if insert is None:
        raise ConversionError(f"cannot convert unit '{unit_name}' to metres")
    factor = float(ez_units.conversion_factor(insert, InsertUnits.Meters))
    if factor <= 0:
        raise ConversionError(f"invalid scale factor for unit '{unit_name}'")
    return factor


def cad_unit_from_insunits(insunits: int | None) -> str | None:
    if insunits is None:
        return None
    if insunits in (0, int(InsertUnits.Unitless)):
        return None
    return INSUNITS_TO_NAME.get(int(insunits))


def resolve_units(
    *,
    explicit_unit: str | None,
    cad_insunits: int | None,
    settings: Settings,
) -> UnitResolution:
    cad_name = cad_unit_from_insunits(cad_insunits)
    if explicit_unit:
        name = normalize_unit_name(explicit_unit)
        source = UnitSource.EXPLICIT
    elif cad_name:
        name = cad_name
        source = UnitSource.CAD_METADATA
    else:
        name = normalize_unit_name(settings.default_unit)
        source = UnitSource.FALLBACK
    try:
        scale = metres_factor(name)
    except ConversionError as exc:
        raise ConversionError(str(exc)) from exc
    return UnitResolution(
        unit="m",
        unit_source=source,
        scale_factor=scale,
        cad_insunits=cad_insunits,
        original_unit=name,
    )
