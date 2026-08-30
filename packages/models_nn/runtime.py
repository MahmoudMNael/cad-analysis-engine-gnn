"""Unified runtime. Downstream always calls predict(); heuristic vs GNN is internal."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from packages.common.configuration import Settings
from packages.common.logging import get_logger
from packages.contracts.enums import DecisionSource
from packages.ingest.models import Primitive
from packages.models_nn.common import FACE_CLASSES, EntityScore, FaceScore, GapScore
from packages.models_nn.heuristics import (
    HeuristicEntityClassifier,
    HeuristicFaceClassifier,
    HeuristicGapClassifier,
)
from packages.models_nn.registry import is_gnn_ref, uri_to_path
from packages.primitives.features import EntityFeatures

log = get_logger("models_nn.runtime")


class EntityRuntime:
    def __init__(self, model_ref: str) -> None:
        self.model_ref = model_ref
        self._heuristic = HeuristicEntityClassifier()

    def predict(self, primitive: Primitive, features: EntityFeatures, settings: Settings) -> EntityScore:
        return self._heuristic.predict(primitive, features, settings)


class GapRuntime:
    def __init__(self, model_ref: str) -> None:
        self.model_ref = model_ref
        self._heuristic = HeuristicGapClassifier()

    def predict(self, *args, **kwargs) -> GapScore:
        return self._heuristic.predict(*args, **kwargs)


class FaceRuntime:
    def __init__(self, model_ref: str, checkpoint_uri: str | None = None) -> None:
        self.model_ref = model_ref
        self._heuristic = HeuristicFaceClassifier()
        self._gnn = None
        self._checkpoint_uri = checkpoint_uri
        if is_gnn_ref(model_ref):
            self._try_load_gnn(checkpoint_uri)

    @property
    def decision_source_name(self) -> str:
        return "model" if self._gnn is not None else "heuristic"

    def predict(self, face_id: str, features: dict[str, float], settings: Settings) -> FaceScore:
        scores = self.predict_graph([face_id], {face_id: features}, [], [], settings)
        return scores[face_id]

    def predict_graph(
        self,
        face_ids: Sequence[str],
        features_by_id: Mapping[str, dict[str, float]],
        edge_index: Sequence[tuple[int, int]],
        edge_attr: Sequence[Sequence[float]],
        settings: Settings,
    ) -> dict[str, FaceScore]:
        heuristic_scores = {
            face_id: self._heuristic.predict(face_id, dict(features_by_id.get(face_id) or {}), settings)
            for face_id in face_ids
        }
        if self._gnn is None or not face_ids:
            return heuristic_scores
        try:
            from packages.models_nn.features.face import vectorize_face_features

            node_features = [vectorize_face_features(features_by_id.get(face_id) or {}) for face_id in face_ids]
            attr = [list(row) for row in edge_attr]
            probas = self._gnn.predict_proba(node_features, list(edge_index), attr)
        except Exception as exc:
            log.warning(
                "face_gnn_failed_fallback_heuristic",
                extra={
                    "stage": "face_classification",
                    "model_version": self.model_ref,
                    "error_detail": str(exc)[:500],
                },
            )
            return heuristic_scores
        out: dict[str, FaceScore] = {}
        for i, face_id in enumerate(face_ids):
            probs = {name: float(probas[i].get(name, 0.0)) for name in FACE_CLASSES}
            predicted = max(probs, key=probs.get)
            confidence = float(probs[predicted])
            gnn_entropy = _entropy(probs)
            if confidence < settings.face_threshold:
                fallback = heuristic_scores[face_id]
                out[face_id] = FaceScore(
                    face_id=face_id,
                    probabilities=fallback.probabilities,
                    predicted_class=fallback.predicted_class,
                    confidence=fallback.confidence,
                    entropy=gnn_entropy,
                    model_version=self.model_ref,
                    decision_source=DecisionSource.HEURISTIC,
                    explanation=(
                        f"gnn uncertain (conf={confidence:.3f} entropy={gnn_entropy:.3f}); "
                        f"fallback {fallback.explanation}"
                    ),
                )
                continue
            out[face_id] = FaceScore(
                face_id=face_id,
                probabilities=probs,
                predicted_class=predicted,
                confidence=confidence,
                entropy=gnn_entropy,
                model_version=self.model_ref,
                decision_source=DecisionSource.MODEL,
                explanation=f"gnn face class; argmax={predicted} conf={confidence:.3f}",
            )
        return out

    def _try_load_gnn(self, checkpoint_uri: str | None) -> None:
        if not checkpoint_uri:
            log.warning(
                "face_gnn_checkpoint_missing",
                extra={"stage": "face_classification", "model_version": self.model_ref},
            )
            return
        try:
            from packages.models_nn.face_gnn import FaceGNN

            path = uri_to_path(checkpoint_uri)
            self._gnn = FaceGNN.load(path)
        except Exception as exc:
            log.warning(
                "face_gnn_load_failed",
                extra={
                    "stage": "face_classification",
                    "model_version": self.model_ref,
                    "error_detail": str(exc)[:500],
                },
            )
            self._gnn = None


def _entropy(probs: dict[str, float]) -> float:
    from packages.models_nn.common import entropy

    return entropy(probs)
