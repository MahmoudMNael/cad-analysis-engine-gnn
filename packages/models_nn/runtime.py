"""Unified runtime. Downstream always calls predict(); heuristic vs GNN is internal."""

from __future__ import annotations

from packages.common.configuration import Settings
from packages.ingest.models import Primitive
from packages.models_nn.common import EntityScore, FaceScore, GapScore
from packages.models_nn.heuristics import (
    HeuristicEntityClassifier,
    HeuristicFaceClassifier,
    HeuristicGapClassifier,
)
from packages.primitives.features import EntityFeatures


class EntityRuntime:
    def __init__(self, model_ref: str) -> None:
        self.model_ref = model_ref
        self._heuristic = HeuristicEntityClassifier()

    def predict(self, primitive: Primitive, features: EntityFeatures, settings: Settings) -> EntityScore:
        if self.model_ref.startswith("heuristic:"):
            return self._heuristic.predict(primitive, features, settings)
        return self._heuristic.predict(primitive, features, settings)


class GapRuntime:
    def __init__(self, model_ref: str) -> None:
        self.model_ref = model_ref
        self._heuristic = HeuristicGapClassifier()

    def predict(self, *args, **kwargs) -> GapScore:
        return self._heuristic.predict(*args, **kwargs)


class FaceRuntime:
    def __init__(self, model_ref: str) -> None:
        self.model_ref = model_ref
        self._heuristic = HeuristicFaceClassifier()

    def predict(self, face_id: str, features: dict[str, float], settings: Settings) -> FaceScore:
        return self._heuristic.predict(face_id, features, settings)
