from __future__ import annotations

FACE_FEATURE_SCHEMA_VERSION = "face_features_v1"


def extract_face_feature_vector(features: dict[str, float]) -> dict[str, float]:
    return dict(features)
