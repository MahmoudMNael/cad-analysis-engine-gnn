"""DXF load, paperspace exclusion, primitive extraction, unit conversion."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import ezdxf
from ezdxf import recover
from ezdxf.entities import DXFEntity
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry

from packages.common.configuration import Settings
from packages.common.errors import CadParseError, ConversionError, UnsupportedFormatError
from packages.common.geometry import finite_geometry, scale_geometry
from packages.contracts.dto.common import WarningDto
from packages.contracts.enums import CadFormat, PrimitiveKind, WarningCode, WarningSeverity
from packages.ingest.blocks import flatten_insert
from packages.ingest.models import BBox, Document, Primitive, bbox_from_geom, primitive_from_parts
from packages.ingest.normalization import normalize_geometry
from packages.ingest.units import resolve_units
from packages.primitives.ids import assign_primitive_id

SUPPORTED_EXTENSIONS = {".dxf"}
KIND_BY_DXFTYPE = {
    "LINE": PrimitiveKind.LINE,
    "ARC": PrimitiveKind.ARC,
    "CIRCLE": PrimitiveKind.CIRCLE,
    "POLYLINE": PrimitiveKind.POLYLINE,
    "LWPOLYLINE": PrimitiveKind.LWPOLYLINE,
    "HATCH": PrimitiveKind.HATCH,
    "TEXT": PrimitiveKind.TEXT,
    "MTEXT": PrimitiveKind.TEXT,
    "ATTRIB": PrimitiveKind.TEXT,
    "ATTDEF": PrimitiveKind.TEXT,
    "INSERT": PrimitiveKind.BLOCK,
    "SPLINE": PrimitiveKind.POLYLINE,
    "ELLIPSE": PrimitiveKind.ARC,
}


def sniff_format(filename: str, header: bytes) -> CadFormat:
    lower = filename.lower()
    if lower.endswith(".dwg") or header.startswith(b"AC10"):
        raise UnsupportedFormatError("DWG is not supported; upload DXF")
    if lower.endswith(".dxf"):
        return CadFormat.DXF
    text = header.decode("latin-1", errors="ignore")
    if "SECTION" in text or "ENTITIES" in text:
        return CadFormat.DXF
    raise UnsupportedFormatError("unsupported CAD format; expected DXF")


def load_dxf(path: Path) -> tuple[Any, list[WarningDto]]:
    warnings: list[WarningDto] = []
    try:
        doc, auditor = recover.readfile(str(path))
    except Exception as exc:
        raise CadParseError(f"failed to parse DXF: {exc}") from exc
    if auditor.has_errors:
        warnings.append(
            WarningDto(
                code=WarningCode.CAD_PARSE_WARNING,
                severity=WarningSeverity.WARNING,
                message="DXF recovered with auditor errors",
                target_ids=[],
            )
        )
    return doc, warnings


def _entity_layer(entity: DXFEntity) -> str:
    try:
        return str(entity.dxf.layer)
    except Exception:
        return "0"


def _entity_color(entity: DXFEntity) -> int | None:
    try:
        return int(entity.dxf.color)
    except Exception:
        return None


def _entity_linetype(entity: DXFEntity) -> str | None:
    try:
        return str(entity.dxf.linetype)
    except Exception:
        return None


def _handle(entity: DXFEntity) -> str | None:
    try:
        value = entity.dxf.handle
    except Exception:
        return None
    return str(value) if value is not None else None


def _flatten_points(entity: DXFEntity) -> list[tuple[float, float]]:
    if hasattr(entity, "flattening"):
        try:
            return [(float(p.x), float(p.y)) for p in entity.flattening(0.01)]
        except Exception:
            pass
    return []


def geometry_from_entity(entity: DXFEntity) -> tuple[BaseGeometry | None, dict[str, Any]]:
    dxftype = entity.dxftype()
    attrs: dict[str, Any] = {}
    try:
        if dxftype == "LINE":
            start, end = entity.dxf.start, entity.dxf.end
            return LineString([(float(start.x), float(start.y)), (float(end.x), float(end.y))]), attrs
        if dxftype == "LWPOLYLINE":
            points = [(float(p[0]), float(p[1])) for p in entity.get_points("xy")]
            if len(points) < 2:
                return None, attrs
            if entity.closed and points[0] != points[-1]:
                points = points + [points[0]]
            attrs["closed"] = bool(entity.closed)
            return LineString(points), attrs
        if dxftype == "POLYLINE":
            points = [(float(v.dxf.location.x), float(v.dxf.location.y)) for v in entity.vertices]
            if len(points) < 2:
                return None, attrs
            if entity.is_closed and points[0] != points[-1]:
                points = points + [points[0]]
            attrs["closed"] = bool(entity.is_closed)
            return LineString(points), attrs
        if dxftype == "ARC":
            center = entity.dxf.center
            attrs.update(
                {
                    "center": (float(center.x), float(center.y)),
                    "radius": float(entity.dxf.radius),
                    "start_angle": float(entity.dxf.start_angle),
                    "end_angle": float(entity.dxf.end_angle),
                }
            )
            pts = _flatten_points(entity)
            if len(pts) >= 2:
                return LineString(pts), attrs
            return None, attrs
        if dxftype == "CIRCLE":
            center = entity.dxf.center
            radius = float(entity.dxf.radius)
            attrs.update({"center": (float(center.x), float(center.y)), "radius": radius})
            pts = _flatten_points(entity)
            if len(pts) >= 3:
                if pts[0] != pts[-1]:
                    pts.append(pts[0])
                return LineString(pts), attrs
            return Point(float(center.x), float(center.y)).buffer(radius).boundary, attrs
        if dxftype == "HATCH":
            paths = []
            for path in entity.paths:
                verts = []
                if hasattr(path, "vertices"):
                    verts = [(float(v[0]), float(v[1])) for v in path.vertices]
                if len(verts) >= 3:
                    if verts[0] != verts[-1]:
                        verts.append(verts[0])
                    paths.append(verts)
            if not paths:
                return None, attrs
            try:
                holes = [p for p in paths[1:]] if len(paths) > 1 else None
                return Polygon(paths[0], holes), attrs
            except Exception:
                return LineString(paths[0]), attrs
        if dxftype in {"TEXT", "ATTRIB", "ATTDEF"}:
            insert = entity.dxf.insert
            height = float(getattr(entity.dxf, "height", 1.0) or 1.0)
            rotation = float(getattr(entity.dxf, "rotation", 0.0) or 0.0)
            content = str(entity.dxf.text) if hasattr(entity.dxf, "text") else ""
            attrs.update({"content": content, "rotation": rotation, "height": height})
            x, y = float(insert.x), float(insert.y)
            width = max(height * max(len(content), 1) * 0.6, height)
            return Point(x, y).buffer(max(height, 0.01)).envelope, attrs
        if dxftype == "MTEXT":
            insert = entity.dxf.insert
            content = entity.text if hasattr(entity, "text") else ""
            height = float(getattr(entity.dxf, "char_height", 1.0) or 1.0)
            rotation = float(getattr(entity.dxf, "rotation", 0.0) or 0.0)
            attrs.update({"content": str(content), "rotation": rotation, "height": height})
            x, y = float(insert.x), float(insert.y)
            return Point(x, y).buffer(max(height, 0.01)).envelope, attrs
        if dxftype in {"SPLINE", "ELLIPSE"}:
            pts = _flatten_points(entity)
            if len(pts) >= 2:
                return LineString(pts), attrs
            return None, attrs
        if dxftype in {"DIMENSION", "ALIGNED_DIMENSION"}:
            attrs["measurement"] = True
            loc = getattr(entity.dxf, "defpoint", None) or getattr(entity.dxf, "text_midpoint", None)
            if loc is not None:
                return Point(float(loc.x), float(loc.y)), attrs
            return None, attrs
    except Exception:
        return None, attrs
    return None, attrs


def iter_modelspace_entities(doc: Any) -> list[tuple[DXFEntity, str | None]]:
    msp = doc.modelspace()
    out: list[tuple[DXFEntity, str | None]] = []
    for entity in msp:
        if entity.dxftype() == "INSERT":
            flattened = list(flatten_insert(entity))
            if flattened:
                out.extend(flattened)
            else:
                out.append((entity, str(entity.dxf.name)))
        else:
            out.append((entity, None))
    return out


def build_document(
    path: Path,
    source_hash: str,
    settings: Settings,
    explicit_unit: str | None = None,
) -> tuple[Document, list[WarningDto]]:
    fmt = sniff_format(path.name, path.read_bytes()[:256])
    if fmt is not CadFormat.DXF:
        raise UnsupportedFormatError("unsupported CAD format")
    doc, warnings = load_dxf(path)
    try:
        insunits = int(doc.units)
    except Exception:
        insunits = 0
    resolution = resolve_units(
        explicit_unit=explicit_unit,
        cad_insunits=insunits,
        settings=settings,
    )
    extracted: list[Primitive] = []
    for entity, block_name in iter_modelspace_entities(doc):
        dxftype = entity.dxftype()
        if dxftype in {"VIEWPORT", "XLINE", "RAY", "IMAGE", "UNDERLAY", "OLE2FRAME"}:
            continue
        geom, attrs = geometry_from_entity(entity)
        if geom is None or geom.is_empty:
            continue
        geom = scale_geometry(geom, resolution.scale_factor)
        geom = normalize_geometry(geom, settings.coord_precision)
        if geom is None or geom.is_empty or not finite_geometry(geom):
            continue
        if dxftype in {"DIMENSION", "ALIGNED_DIMENSION"}:
            attrs["measurement"] = True
        kind = KIND_BY_DXFTYPE.get(dxftype)
        if kind is None:
            if attrs.get("measurement"):
                kind = PrimitiveKind.LINE
            elif geom.geom_type in {"LineString", "MultiLineString"}:
                kind = PrimitiveKind.LINE
            elif geom.geom_type == "Polygon":
                kind = PrimitiveKind.HATCH
            else:
                continue
        primitive = primitive_from_parts(
            primitive_id="pending",
            kind=kind,
            geometry=geom,
            layer=_entity_layer(entity),
            color=_entity_color(entity),
            linetype=_entity_linetype(entity),
            source_entity_type=dxftype,
            source_handle=_handle(entity),
            block_name=block_name,
            attributes=attrs,
        )
        primitive_id = assign_primitive_id(source_hash, primitive, settings.coord_precision)
        extracted.append(
            primitive_from_parts(
                primitive_id=primitive_id,
                kind=primitive.kind,
                geometry=primitive.geometry,
                layer=primitive.layer,
                color=primitive.color,
                linetype=primitive.linetype,
                source_entity_type=primitive.source_entity_type,
                source_handle=primitive.source_handle,
                block_name=primitive.block_name,
                attributes=primitive.attributes,
            )
        )
    if not extracted:
        raise CadParseError("DXF contained no usable modelspace primitives")

    minx = min(p.bbox.minx for p in extracted)
    miny = min(p.bbox.miny for p in extracted)
    maxx = max(p.bbox.maxx for p in extracted)
    maxy = max(p.bbox.maxy for p in extracted)
    bbox = BBox(minx, miny, maxx, maxy)
    extent = bbox.extent()
    if not math.isfinite(extent):
        raise ConversionError("converted coordinates are not finite")
    if extent > settings.max_bbox_extent_m:
        raise ConversionError("converted drawing extent is impossibly large")
    if extent < settings.min_bbox_extent_m:
        raise ConversionError("converted drawing extent is impossibly small")

    metadata = {
        "scale_factor": resolution.scale_factor,
        "original_unit": resolution.original_unit,
        "cad_insunits": resolution.cad_insunits,
        "source_filename": path.name,
        "primitive_count": len(extracted),
    }
    document = Document(
        document_id=source_hash,
        source_hash=source_hash,
        format=fmt,
        unit="m",
        unit_source=resolution.unit_source,
        bbox=bbox,
        primitives=tuple(extracted),
        metadata=metadata,
    )
    return document, warnings
