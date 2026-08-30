from __future__ import annotations

import math
from dataclasses import dataclass

from packages.contracts.enums import DecisionSource, EntityClass, FaceClass, GapDecision

ENTITY_CLASSES: tuple[str, ...] = tuple(c.value for c in EntityClass)
FACE_CLASSES: tuple[str, ...] = tuple(c.value for c in FaceClass)
FACE_CLASS_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(FACE_CLASSES)}
INDEX_TO_FACE_CLASS: tuple[str, ...] = FACE_CLASSES


def entropy(probs: dict[str, float]) -> float:
    total = 0.0
    for p in probs.values():
        if p > 0:
            total -= p * math.log(p + 1e-12)
    return float(total)


def softmax(scores: dict[str, float]) -> dict[str, float]:
    m = max(scores.values())
    exps = {k: math.exp(v - m) for k, v in scores.items()}
    z = sum(exps.values()) or 1.0
    return {k: v / z for k, v in exps.items()}


@dataclass(frozen=True)
class EntityScore:
    target_id: str
    probabilities: dict[str, float]
    predicted_class: str
    confidence: float
    entropy: float
    model_version: str
    decision_source: DecisionSource
    explanation: str


@dataclass(frozen=True)
class GapScore:
    pair_id: str
    probability: float
    decision: GapDecision
    confidence: float
    model_version: str
    decision_source: DecisionSource
    explanation: str


@dataclass(frozen=True)
class FaceScore:
    face_id: str
    probabilities: dict[str, float]
    predicted_class: str
    confidence: float
    entropy: float
    model_version: str
    decision_source: DecisionSource
    explanation: str
