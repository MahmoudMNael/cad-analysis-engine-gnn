"""DWG to DXF conversion via ODA File Converter."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from packages.common.configuration import Settings
from packages.common.errors import ConversionError

DEFAULT_ODA_PATH = "/usr/bin/ODAFileConverter_27.1.0.0/ODAFileConverter"


def is_dwg(filename: str, header: bytes) -> bool:
    lower = filename.lower()
    return lower.endswith(".dwg") or header.startswith(b"AC10")


def _oda_exec_path(settings: Settings) -> str:
    path = settings.oda_converter_path.strip()
    if path and Path(path).is_file():
        return path
    if Path(DEFAULT_ODA_PATH).is_file():
        return DEFAULT_ODA_PATH
    found = shutil.which("ODAFileConverter")
    if found:
        return found
    raise ConversionError(
        "ODA File Converter not found; install ODA or set CAD_ODA_CONVERTER_PATH"
    )


def _oda_available(settings: Settings) -> bool:
    path = settings.oda_converter_path.strip()
    if path and Path(path).is_file():
        return True
    if Path(DEFAULT_ODA_PATH).is_file():
        return True
    return shutil.which("ODAFileConverter") is not None


def _map_version(version: str) -> str:
    normalized = version.strip().upper()
    if normalized.startswith("ACAD"):
        return normalized
    if normalized.startswith("R") and normalized[1:].isdigit():
        return f"ACAD{normalized[1:]}"
    return normalized


def convert_dwg_to_dxf(source: Path, dest: Path, settings: Settings) -> None:
    if not _oda_available(settings):
        raise ConversionError(
            "ODA File Converter not found; install ODA or set CAD_ODA_CONVERTER_PATH"
        )

    source = source.resolve()
    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    oda_fc = _oda_exec_path(settings)
    version = _map_version(settings.oda_dxf_version)

    with tempfile.TemporaryDirectory(prefix="odafc_") as tmp_dir:
        args = [
            str(source.parent),
            tmp_dir,
            version,
            "DXF",
            "0",
            "1",
            source.name,
        ]
        proc = subprocess.run([oda_fc, *args], capture_output=True, text=True)
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip() or "unknown error"
            raise ConversionError(
                f"DWG to DXF conversion failed: return code {proc.returncode}. {detail}"
            )
        results = list(Path(tmp_dir).iterdir())
        if not results:
            raise ConversionError("DWG to DXF conversion produced no output file")
        shutil.move(str(results[0]), str(dest))

    if not dest.is_file() or dest.stat().st_size == 0:
        raise ConversionError("DWG to DXF conversion produced empty output")
