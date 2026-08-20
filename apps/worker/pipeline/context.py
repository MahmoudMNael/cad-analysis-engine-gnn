from __future__ import annotations

import inspect
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from packages.common.errors import AppError
from packages.common.hashing import sha256_file
from packages.common.logging import get_logger
from packages.common.timing import TimingStore
from packages.contracts.dto.common import WarningDto
from packages.contracts.enums import ErrorCode


class PipelineBlock(Protocol):
    name: str
    input_schema: type
    output_schema: type

    def run(self, input: Any, context: PipelineContext) -> Any: ...


@dataclass
class _NamedBlock:
    name: str
    version: str
    model_ref: str
    input_schema: type = dict
    output_schema: type = dict


@dataclass
class StageTracker:
    count_out: int = 0


@dataclass
class PipelineContext:
    job_id: str
    engine_sha: str
    configuration: Any
    model_versions: dict[str, str]
    source_hash: str
    timings: TimingStore = field(default_factory=TimingStore)
    warnings: list[WarningDto] = field(default_factory=list)
    logger: Any = field(default_factory=lambda: get_logger("pipeline"))
    block_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    worker_id: str | None = None
    current_stage: str | None = None

    def _extra(self, **fields: Any) -> dict[str, Any]:
        payload = {"job_id": self.job_id}
        if self.worker_id:
            payload["worker_id"] = self.worker_id
        payload.update(fields)
        return payload

    def add_warnings(self, items: list[WarningDto]) -> None:
        self.warnings.extend(items)

    @contextmanager
    def track_stage(
        self,
        name: str,
        *,
        model_ref: str,
        count_in: int = 0,
        version: str = "1",
    ) -> Iterator[StageTracker]:
        self.current_stage = name
        self.logger.info(
            "stage_start",
            extra=self._extra(
                stage=name,
                block=name,
                count_in=count_in,
                model_version=model_ref,
            ),
        )
        started = time.perf_counter()
        tracker = StageTracker(count_out=count_in)
        try:
            yield tracker
        except Exception as exc:
            duration_ms = round((time.perf_counter() - started) * 1000.0, 3)
            code = exc.code.value if isinstance(exc, AppError) else ErrorCode.INTERNAL_ERROR.value
            self.logger.exception(
                "stage_failed",
                extra=self._extra(
                    stage=name,
                    block=name,
                    duration_ms=duration_ms,
                    count_in=count_in,
                    model_version=model_ref,
                    error_code=code,
                    error_detail=str(exc)[:500],
                ),
            )
            raise
        duration_ms = (time.perf_counter() - started) * 1000.0
        self.timings.add(name, duration_ms)
        self.record_block(
            _NamedBlock(name=name, version=version, model_ref=model_ref),
            duration_ms=duration_ms,
            count_in=count_in,
            count_out=tracker.count_out,
            model_ref=model_ref,
        )

    def record_block(
        self,
        block: Any,
        *,
        duration_ms: float,
        count_in: int,
        count_out: int,
        model_ref: str,
    ) -> None:
        impl = "unknown"
        try:
            impl = sha256_file(Path(inspect.getfile(block.__class__)))
        except (OSError, TypeError):
            pass
        self.block_meta[block.name] = {
            "name": block.name,
            "version": getattr(block, "version", "1"),
            "implementation_sha": impl,
            "model_ref": model_ref,
            "duration_ms": duration_ms,
            "count_in": count_in,
            "count_out": count_out,
        }
        self.logger.info(
            "stage_complete",
            extra=self._extra(
                stage=block.name,
                block=block.name,
                duration_ms=round(duration_ms, 3),
                count_in=count_in,
                count_out=count_out,
                model_version=model_ref,
            ),
        )
