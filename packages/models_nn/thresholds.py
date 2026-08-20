from __future__ import annotations

from packages.common.configuration import Settings


class Thresholds:
    def __init__(self, settings: Settings) -> None:
        self.entity = settings.entity_threshold
        self.gap = settings.gap_threshold
        self.face = settings.face_threshold
        self.gap_max_distance_m = settings.gap_max_distance_m
        self.face_min_area_m2 = settings.face_min_area_m2
