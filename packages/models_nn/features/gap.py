from __future__ import annotations

GAP_FEATURE_SCHEMA_VERSION = "gap_features_v1"


def extract_gap_features(
    *,
    distance_m: float,
    d_max: float,
    collinear: bool,
    already_connected: bool,
    crosses: bool,
) -> dict[str, float]:
    return {
        "distance_m": float(distance_m),
        "distance_norm": float(min(distance_m / max(d_max, 1e-9), 2.0)),
        "collinear": 1.0 if collinear else 0.0,
        "already_connected": 1.0 if already_connected else 0.0,
        "crosses": 1.0 if crosses else 0.0,
    }
