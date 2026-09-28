from pathlib import Path

from packages.common.configuration import Settings
from tests.helpers.dxf_builder import write_simple_dxf
from tests.helpers.pipeline import run_dxf
from tests.helpers.regression import pipeline_fingerprint
from tests.helpers.synthetic_cases import build_two_adjacent_rooms


def test_entity_classify_workers_match_sequential(tmp_path: Path):
    path = write_simple_dxf(tmp_path / "two.dxf", build_two_adjacent_rooms)
    sequential = run_dxf(path, settings=Settings(entity_classify_workers=1), job_id="w1")
    parallel = run_dxf(path, settings=Settings(entity_classify_workers=4), job_id="w4")
    assert sequential.plan_graph_payload["entity_scores"] == parallel.plan_graph_payload["entity_scores"]
    assert pipeline_fingerprint(sequential) == pipeline_fingerprint(parallel)
