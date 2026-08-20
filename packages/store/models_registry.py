"""Resolve pinned model versions for a job. A later publish cannot change them."""

from __future__ import annotations

from dataclasses import dataclass

from packages.store.db.repositories.models import ModelRegistryRepository

HEURISTIC_DEFAULTS = {
    "entity": "heuristic:entity_v1",
    "gap": "heuristic:gap_v1",
    "face": "heuristic:face_v1",
}

FEATURE_SCHEMA = {
    "entity": "entity_features_v1",
    "gap": "gap_features_v1",
    "face": "face_features_v1",
}


@dataclass(frozen=True)
class PinnedModels:
    entity_model_version: str
    gap_model_version: str
    face_model_version: str

    def as_dict(self) -> dict[str, str]:
        return {
            "entity_model_version": self.entity_model_version,
            "gap_model_version": self.gap_model_version,
            "face_model_version": self.face_model_version,
        }


def seed_heuristic_registry(repo: ModelRegistryRepository) -> None:
    for name, version in HEURISTIC_DEFAULTS.items():
        repo.ensure_heuristic(name, version, FEATURE_SCHEMA[name])


def resolve_pinned_models(repo: ModelRegistryRepository) -> PinnedModels:
    seed_heuristic_registry(repo)
    entity = repo.promoted_version("entity") or HEURISTIC_DEFAULTS["entity"]
    gap = repo.promoted_version("gap") or HEURISTIC_DEFAULTS["gap"]
    face = repo.promoted_version("face") or HEURISTIC_DEFAULTS["face"]
    return PinnedModels(
        entity_model_version=entity,
        gap_model_version=gap,
        face_model_version=face,
    )
