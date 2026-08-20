"""Typed process configuration. Every job-affecting value is part of configuration_hash."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CAD_",
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    database_url: str = "postgresql://cad:cad@localhost:5432/cad"
    redis_url: str = "redis://localhost:6379/0"
    redis_queue_key: str = "cad:jobs"
    object_store_dir: Path = Path("./object-storage")

    max_upload_bytes: int = 50 * 1024 * 1024
    default_unit: str = "mm"
    snap_tolerance_m: float = 0.02
    intersection_tolerance_m: float = 0.001
    gap_max_distance_m: float = 2.0
    gap_threshold: float = 0.5
    face_min_area_m2: float = 0.25
    entity_threshold: float = 0.35
    face_threshold: float = 0.35
    object_assignment_tolerance_m: float = 0.05
    job_timeout_seconds: int = 300
    retrain_event_threshold: int = 2000

    coord_precision: int = 6
    min_segment_length_m: float = 0.005
    max_bbox_extent_m: float = 100_000.0
    min_bbox_extent_m: float = 0.01
    outdoor_area_m2: float = 250.0
    cavity_area_m2: float = 0.8
    shaft_area_m2: float = 2.5
    door_width_min_m: float = 0.55
    door_width_max_m: float = 1.4
    window_width_min_m: float = 0.35
    window_width_max_m: float = 3.0
    pipeline_version: str = "1.0.0"
    layout_schema: str = "layout.v1"
    allowed_source_url_hosts: str = ""
    engine_sha_override: str = ""

    def allowed_hosts(self) -> tuple[str, ...]:
        raw = self.allowed_source_url_hosts.strip()
        if not raw:
            return ()
        return tuple(part.strip() for part in raw.split(",") if part.strip())

    def job_config_dict(self) -> dict[str, float | int | str]:
        return {
            "MAX_UPLOAD_BYTES": self.max_upload_bytes,
            "DEFAULT_UNIT": self.default_unit,
            "SNAP_TOLERANCE_M": self.snap_tolerance_m,
            "INTERSECTION_TOLERANCE_M": self.intersection_tolerance_m,
            "GAP_MAX_DISTANCE_M": self.gap_max_distance_m,
            "GAP_THRESHOLD": self.gap_threshold,
            "FACE_MIN_AREA_M2": self.face_min_area_m2,
            "ENTITY_THRESHOLD": self.entity_threshold,
            "FACE_THRESHOLD": self.face_threshold,
            "OBJECT_ASSIGNMENT_TOLERANCE_M": self.object_assignment_tolerance_m,
            "JOB_TIMEOUT_SECONDS": self.job_timeout_seconds,
            "RETRAIN_EVENT_THRESHOLD": self.retrain_event_threshold,
            "COORD_PRECISION": self.coord_precision,
            "MIN_SEGMENT_LENGTH_M": self.min_segment_length_m,
            "MAX_BBOX_EXTENT_M": self.max_bbox_extent_m,
            "MIN_BBOX_EXTENT_M": self.min_bbox_extent_m,
            "OUTDOOR_AREA_M2": self.outdoor_area_m2,
            "CAVITY_AREA_M2": self.cavity_area_m2,
            "SHAFT_AREA_M2": self.shaft_area_m2,
            "DOOR_WIDTH_MIN_M": self.door_width_min_m,
            "DOOR_WIDTH_MAX_M": self.door_width_max_m,
            "WINDOW_WIDTH_MIN_M": self.window_width_min_m,
            "WINDOW_WIDTH_MAX_M": self.window_width_max_m,
            "PIPELINE_VERSION": self.pipeline_version,
            "LAYOUT_SCHEMA": self.layout_schema,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
