from packages.common.configuration import Settings
from packages.eval.promotion import promotion_gate
from packages.models_nn.features.face import FACE_FEATURE_SCHEMA_VERSION


def test_promotion_requires_held_out_and_beating_production():
    settings = Settings(face_promote_min_macro_f1=0.0, face_promote_tolerance=0.02)
    rejected = promotion_gate(
        candidate_metrics={"macro_f1": 0.4, "f1_indoor": 0.4, "delete_rate_proxy": 0.2, "held_out_passed": 1.0},
        production_metrics={"macro_f1": 0.5, "f1_indoor": 0.5, "delete_rate_proxy": 0.1},
        feature_schema_version=FACE_FEATURE_SCHEMA_VERSION,
        artifact_ok=True,
        settings=settings,
    )
    assert rejected.accepted is False
    assert any("does not beat production" in reason for reason in rejected.reasons)

    missing_held_out = promotion_gate(
        candidate_metrics={"macro_f1": 0.9, "held_out_passed": 0.0},
        production_metrics={"macro_f1": 0.1},
        feature_schema_version=FACE_FEATURE_SCHEMA_VERSION,
        artifact_ok=True,
        settings=settings,
    )
    assert missing_held_out.accepted is False

    bad_artifact = promotion_gate(
        candidate_metrics={"macro_f1": 0.9, "held_out_passed": 1.0},
        production_metrics={"macro_f1": 0.1},
        feature_schema_version=FACE_FEATURE_SCHEMA_VERSION,
        artifact_ok=False,
        settings=settings,
    )
    assert bad_artifact.accepted is False

    accepted = promotion_gate(
        candidate_metrics={
            "macro_f1": 0.8,
            "f1_indoor": 0.85,
            "delete_rate_proxy": 0.05,
            "held_out_passed": 1.0,
        },
        production_metrics={"macro_f1": 0.4, "f1_indoor": 0.5, "delete_rate_proxy": 0.1},
        feature_schema_version=FACE_FEATURE_SCHEMA_VERSION,
        artifact_ok=True,
        settings=settings,
    )
    assert accepted.accepted is True
    assert accepted.reasons == ()


def test_protected_metric_regression_blocks_promotion():
    decision = promotion_gate(
        candidate_metrics={
            "macro_f1": 0.9,
            "f1_indoor": 0.2,
            "delete_rate_proxy": 0.05,
            "held_out_passed": 1.0,
        },
        production_metrics={"macro_f1": 0.4, "f1_indoor": 0.8, "delete_rate_proxy": 0.1},
        feature_schema_version=FACE_FEATURE_SCHEMA_VERSION,
        artifact_ok=True,
        settings=Settings(face_promote_tolerance=0.02),
    )
    assert decision.accepted is False
    assert any("f1_indoor" in reason for reason in decision.reasons)
