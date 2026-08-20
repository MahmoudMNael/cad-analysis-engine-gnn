from __future__ import annotations

from pathlib import Path

import ezdxf
from ezdxf.enums import InsertUnits


def write_dxf(path: Path, *, insunits: InsertUnits = InsertUnits.Millimeters) -> ezdxf.document.Drawing:
    doc = ezdxf.new("R2010")
    doc.units = insunits
    doc.saveas(path)
    return ezdxf.readfile(path)


def save_with_entities(path: Path, insunits: InsertUnits = InsertUnits.Millimeters) -> ezdxf.document.Drawing:
    doc = ezdxf.new("R2010")
    doc.units = insunits
    return doc


def add_rect(msp, origin: tuple[float, float], width: float, height: float, layer: str = "WALL") -> None:
    x, y = origin
    pts = [(x, y), (x + width, y), (x + width, y + height), (x, y + height), (x, y)]
    msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": layer})


def add_lines(msp, segments: list[tuple[tuple[float, float], tuple[float, float]]], layer: str = "WALL") -> None:
    for a, b in segments:
        msp.add_line(a, b, dxfattribs={"layer": layer})


def write_simple_dxf(
    path: Path,
    builder,
    insunits: InsertUnits = InsertUnits.Millimeters,
) -> Path:
    doc = ezdxf.new("R2010")
    doc.units = insunits
    builder(doc.modelspace())
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(path)
    return path
