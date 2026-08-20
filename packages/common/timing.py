"""Stage timing helpers."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class TimingRecord:
    stage: str
    duration_ms: float


@dataclass
class TimingStore:
    records: list[TimingRecord] = field(default_factory=list)

    def add(self, stage: str, duration_ms: float) -> None:
        self.records.append(TimingRecord(stage=stage, duration_ms=round(duration_ms, 3)))

    def as_dict(self) -> dict[str, float]:
        return {item.stage: item.duration_ms for item in self.records}


@contextmanager
def timed_stage(store: TimingStore, stage: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        store.add(stage, (time.perf_counter() - started) * 1000.0)
