from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from packages.common.configuration import Settings
from packages.common.errors import ConversionError
from packages.contracts.enums import CadFormat
from packages.ingest.dwg_converter import convert_dwg_to_dxf, is_dwg, _map_version
from packages.ingest.dxf_loader import sniff_format


def test_is_dwg_by_extension():
    assert is_dwg("plan.dwg", b"") is True


def test_is_dwg_by_magic_bytes():
    assert is_dwg("upload.bin", b"AC1027") is True


def test_is_dwg_false_for_dxf():
    assert is_dwg("plan.dxf", b"0\nSECTION\n") is False


def test_sniff_format_returns_dwg():
    assert sniff_format("plan.dwg", b"AC1027xxxx") == CadFormat.DWG


def test_sniff_format_returns_dxf():
    assert sniff_format("plan.dxf", b"0\nSECTION\n") == CadFormat.DXF


def test_map_version():
    assert _map_version("R2018") == "ACAD2018"
    assert _map_version("ACAD2018") == "ACAD2018"


def test_convert_dwg_to_dxf_success(tmp_path, monkeypatch):
    source = tmp_path / "plan.dwg"
    dest = tmp_path / "plan.dxf"
    source.write_bytes(b"AC1027fake")
    out_dir_holder: dict[str, Path] = {}

    def fake_run(cmd, capture_output, text):
        out_dir = Path(cmd[2])
        out_dir_holder["dir"] = out_dir
        (out_dir / "plan.dxf").write_bytes(b"0\nSECTION\n2\nENTITIES\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="harmless qt warning")

    monkeypatch.setattr("packages.ingest.dwg_converter._oda_available", lambda settings: True)
    monkeypatch.setattr("packages.ingest.dwg_converter._oda_exec_path", lambda settings: "/usr/bin/oda")
    with patch("packages.ingest.dwg_converter.subprocess.run", side_effect=fake_run):
        convert_dwg_to_dxf(source, dest, Settings())
    assert dest.read_bytes().startswith(b"0\nSECTION")


def test_convert_dwg_to_dxf_oda_missing(tmp_path, monkeypatch):
    source = tmp_path / "plan.dwg"
    dest = tmp_path / "plan.dxf"
    source.write_bytes(b"AC1027fake")
    monkeypatch.setattr("packages.ingest.dwg_converter._oda_available", lambda settings: False)
    with pytest.raises(ConversionError, match="ODA File Converter not found"):
        convert_dwg_to_dxf(source, dest, Settings())


def test_convert_dwg_to_dxf_failure(tmp_path, monkeypatch):
    source = tmp_path / "plan.dwg"
    dest = tmp_path / "plan.dxf"
    source.write_bytes(b"AC1027fake")
    monkeypatch.setattr("packages.ingest.dwg_converter._oda_available", lambda settings: True)
    monkeypatch.setattr("packages.ingest.dwg_converter._oda_exec_path", lambda settings: "/usr/bin/oda")
    with patch(
        "packages.ingest.dwg_converter.subprocess.run",
        return_value=SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    ):
        with pytest.raises(ConversionError, match="return code 1"):
            convert_dwg_to_dxf(source, dest, Settings())


def test_convert_dwg_to_dxf_no_output(tmp_path, monkeypatch):
    source = tmp_path / "plan.dwg"
    dest = tmp_path / "plan.dxf"
    source.write_bytes(b"AC1027fake")
    monkeypatch.setattr("packages.ingest.dwg_converter._oda_available", lambda settings: True)
    monkeypatch.setattr("packages.ingest.dwg_converter._oda_exec_path", lambda settings: "/usr/bin/oda")
    with patch(
        "packages.ingest.dwg_converter.subprocess.run",
        return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
    ):
        with pytest.raises(ConversionError, match="no output file"):
            convert_dwg_to_dxf(source, dest, Settings())
