from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.helpers.dxf_builder import write_simple_dxf
from tests.helpers.pipeline import run_dxf
from tests.helpers.regression import pipeline_fingerprint
from tests.helpers.synthetic_cases import PIPELINE_CASES

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden" / "pipeline"


@pytest.mark.parametrize("name,builder", PIPELINE_CASES, ids=[name for name, _ in PIPELINE_CASES])
def test_pipeline_fingerprint_matches_golden(name: str, builder, tmp_path: Path):
    path = write_simple_dxf(tmp_path / f"{name}.dxf", builder)
    result = run_dxf(path, job_id=f"golden-{name}")
    actual = pipeline_fingerprint(result)
    golden_path = GOLDEN_DIR / f"{name}.json"
    assert golden_path.is_file(), f"missing golden {golden_path}"
    expected = json.loads(golden_path.read_text(encoding="utf-8"))
    assert actual == expected
