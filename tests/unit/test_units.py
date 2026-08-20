from packages.common.configuration import Settings
from packages.contracts.enums import UnitSource
from packages.ingest.units import metres_factor, resolve_units


def test_mm_to_metres():
    assert abs(metres_factor("mm") - 0.001) < 1e-12


def test_inches_to_metres():
    assert abs(metres_factor("in") - 0.0254) < 1e-12


def test_explicit_unit_overrides_cad_metadata():
    resolved = resolve_units(explicit_unit="m", cad_insunits=4, settings=Settings())
    assert resolved.unit == "m"
    assert resolved.unit_source is UnitSource.EXPLICIT
    assert resolved.scale_factor == 1.0
    assert resolved.original_unit == "m"


def test_cad_metadata_when_trustworthy():
    resolved = resolve_units(explicit_unit=None, cad_insunits=4, settings=Settings())
    assert resolved.unit_source is UnitSource.CAD_METADATA
    assert resolved.original_unit == "mm"
    assert abs(resolved.scale_factor - 0.001) < 1e-12


def test_fallback_when_unitless():
    resolved = resolve_units(explicit_unit=None, cad_insunits=0, settings=Settings())
    assert resolved.unit_source is UnitSource.FALLBACK
    assert resolved.original_unit == "mm"


def test_build_document_explicit_metres_ignores_inch_header(tmp_path):
    from ezdxf.enums import InsertUnits

    from packages.ingest.dxf_loader import build_document
    from tests.helpers.dxf_builder import add_rect, write_simple_dxf

    path = write_simple_dxf(
        tmp_path / "inch.dxf",
        lambda msp: add_rect(msp, (0, 0), 10, 8),
        insunits=InsertUnits.Inches,
    )
    document, _ = build_document(path, "hash", Settings(), explicit_unit="m")
    assert document.unit_source is UnitSource.EXPLICIT
    assert abs(float(document.metadata["scale_factor"]) - 1.0) < 1e-12
    assert abs(document.bbox.maxx - 10.0) < 1e-6
    assert abs(document.bbox.maxy - 8.0) < 1e-6
