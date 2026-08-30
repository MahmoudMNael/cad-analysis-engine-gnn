"""Heuristic classifiers that return the same score shapes as future GNN runtimes."""

from __future__ import annotations

from packages.common.configuration import Settings
from packages.contracts.enums import DecisionSource, EntityClass, FaceClass, GapDecision
from packages.ingest.models import Primitive
from packages.models_nn.common import (
    ENTITY_CLASSES,
    FACE_CLASSES,
    EntityScore,
    FaceScore,
    GapScore,
    entropy,
    softmax,
)
from packages.models_nn.features.gap import extract_gap_features
from packages.primitives.features import EntityFeatures


def _keyword_hits(text: str, words: tuple[str, ...]) -> bool:
    blob = text.lower()
    return any(word in blob for word in words)


class HeuristicEntityClassifier:
    version = "heuristic:entity_v1"

    def predict(self, primitive: Primitive, features: EntityFeatures, settings: Settings) -> EntityScore:
        scores = {name: 0.05 for name in ENTITY_CLASSES}
        blob = " ".join(
            filter(
                None,
                [
                    primitive.layer,
                    primitive.block_name or "",
                    primitive.source_entity_type,
                    primitive.linetype or "",
                ],
            )
        )
        kind = primitive.kind.value
        if kind == "text" or _keyword_hits(blob, ("text", "anno", "label", "mtext")):
            scores[EntityClass.TEXT.value] += 6.0
        if kind == "hatch" or _keyword_hits(blob, ("hatch", "pattern", "solid")):
            scores[EntityClass.HATCH.value] += 5.0
        if primitive.attributes.get("measurement") or _keyword_hits(
            blob, ("dim", "dimension", "dims")
        ):
            scores[EntityClass.DIM.value] += 6.0
        if _keyword_hits(blob, ("door", "window", "opening", "glaz", "wnd")):
            scores[EntityClass.OPENING.value] += 5.0
        if _keyword_hits(blob, ("furn", "fixture", "equip", "chair", "table", "sofa", "bed")):
            scores[EntityClass.FURNITURE.value] += 5.0
        if _keyword_hits(blob, ("wall", "a-wall", "mur", "partition")):
            scores[EntityClass.WALL.value] += 5.0
        if kind in {"line", "lwpolyline", "polyline", "arc"}:
            scores[EntityClass.WALL.value] += 1.8
            if features.values["length"] > 0.4:
                scores[EntityClass.WALL.value] += 1.2
            if features.values["curvature"] > 0.25 and 0.4 <= features.values["length"] <= 2.5:
                scores[EntityClass.OPENING.value] += 2.5
        if kind == "circle" and features.values["area"] < 1.0:
            scores[EntityClass.FURNITURE.value] += 1.5
        if primitive.block_name and scores[EntityClass.WALL.value] < 3:
            scores[EntityClass.FURNITURE.value] += 1.0

        probs = softmax(scores)
        predicted = max(probs, key=probs.get)
        confidence = float(probs[predicted])
        if confidence < settings.entity_threshold:
            predicted = EntityClass.UNKNOWN.value
        return EntityScore(
            target_id=primitive.id,
            probabilities=probs,
            predicted_class=predicted,
            confidence=confidence,
            entropy=entropy(probs),
            model_version=self.version,
            decision_source=DecisionSource.HEURISTIC,
            explanation=f"heuristic entity scores from layer/geometry; argmax={predicted}",
        )


class HeuristicGapClassifier:
    version = "heuristic:gap_v1"

    def predict(
        self,
        pair_id: str,
        distance_m: float,
        *,
        collinear: bool,
        crosses: bool,
        already_connected: bool,
        settings: Settings,
        opening_like: bool,
    ) -> GapScore:
        features = extract_gap_features(
            distance_m=distance_m,
            d_max=settings.gap_max_distance_m,
            collinear=collinear,
            already_connected=already_connected,
            crosses=crosses,
        )
        probability = max(0.0, 1.0 - features["distance_norm"])
        if collinear:
            probability = min(1.0, probability + 0.15)
        if crosses or already_connected:
            probability = 0.0
        if opening_like:
            decision = GapDecision.PORTAL
            probability = min(probability, 0.45)
            explanation = "gap width looks like a door/window; close as portal not wall"
        elif (
            probability >= settings.gap_threshold
            and distance_m <= settings.gap_max_distance_m
            and not crosses
            and not already_connected
        ):
            decision = GapDecision.CLOSE
            explanation = "short plausible dangling-node gap"
        else:
            decision = GapDecision.REJECT
            explanation = "gap rejected by distance/geometry constraints"
        return GapScore(
            pair_id=pair_id,
            probability=float(probability),
            decision=decision,
            confidence=float(probability if decision != GapDecision.REJECT else 1.0 - probability),
            model_version=self.version,
            decision_source=DecisionSource.HEURISTIC,
            explanation=explanation,
        )


class HeuristicFaceClassifier:
    version = "heuristic:face_v1"

    def predict(self, face_id: str, features: dict[str, float], settings: Settings) -> FaceScore:
        scores = {name: 0.05 for name in FACE_CLASSES}
        area = features.get("area", 0.0)
        compactness = features.get("compactness", 0.0)
        aspect = features.get("aspect_ratio", 1.0)
        doors = features.get("number_of_doors", 0.0)
        windows = features.get("number_of_windows", 0.0)
        unbounded = features.get("adjacent_unbounded", 0.0)
        texts = features.get("text_overlap", 0.0)
        perimeter = features.get("perimeter", 0.0)
        wall_overlap = features.get("wall_overlap", 0.0)
        adjacent = features.get("adjacent_face_count", 0.0)
        wall_density = wall_overlap / max(perimeter, 1e-6)
        portals = doors + windows

        if area < settings.cavity_area_m2:
            scores[FaceClass.CAVITY.value] += 4.0
            if adjacent >= 2:
                scores[FaceClass.CAVITY.value] += 0.6
        elif area < settings.shaft_area_m2 and aspect > 4.0:
            scores[FaceClass.SHAFT.value] += 3.5
        elif (
            unbounded >= 1.0
            and area > settings.outdoor_area_m2
            and portals < 1
            and (compactness < 0.2 or wall_density < 0.08)
        ):
            scores[FaceClass.OUTDOOR.value] += 3.5
        elif portals >= 2 and aspect > 3.0 and compactness < 0.45:
            scores[FaceClass.CORRIDOR.value] += 3.0
        else:
            scores[FaceClass.INDOOR.value] += 3.0
            if texts > 0:
                scores[FaceClass.INDOOR.value] += 0.8
            if compactness > 0.3:
                scores[FaceClass.INDOOR.value] += 0.5
            if wall_density > 0.2:
                scores[FaceClass.INDOOR.value] += 0.4
            if adjacent >= 1 and unbounded < 1:
                scores[FaceClass.INDOOR.value] += 0.4

        probs = softmax(scores)
        predicted = max(probs, key=probs.get)
        confidence = float(probs[predicted])
        if confidence < settings.face_threshold:
            predicted = FaceClass.INDOOR.value
        return FaceScore(
            face_id=face_id,
            probabilities=probs,
            predicted_class=predicted,
            confidence=confidence,
            entropy=entropy(probs),
            model_version=self.version,
            decision_source=DecisionSource.HEURISTIC,
            explanation=f"heuristic face class from area/adjacency/portals; argmax={predicted}",
        )
