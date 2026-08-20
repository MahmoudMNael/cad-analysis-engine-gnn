from __future__ import annotations

from pathlib import Path

from packages.common.configuration import Settings
from packages.common.hashing import configuration_hash, engine_sha_from_sources, sha256_file
from apps.worker.pipeline.context import PipelineContext
from apps.worker.pipeline.runner import PipelineResult, run_pipeline

ROOT = Path(__file__).resolve().parents[2]


def run_dxf(path: Path, settings: Settings | None = None, job_id: str = "test-job") -> PipelineResult:
    cfg = settings or Settings()
    source_hash = sha256_file(path)
    context = PipelineContext(
        job_id=job_id,
        engine_sha=engine_sha_from_sources(ROOT),
        configuration=configuration_hash(cfg.job_config_dict()),
        model_versions={
            "entity_model_version": "heuristic:entity_v1",
            "gap_model_version": "heuristic:gap_v1",
            "face_model_version": "heuristic:face_v1",
        },
        source_hash=source_hash,
    )
    return run_pipeline(source_path=path, context=context, settings=cfg)
