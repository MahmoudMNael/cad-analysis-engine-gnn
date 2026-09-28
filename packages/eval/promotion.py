"""Production promotion gate for GNN-C. Never promote a checkpoint directly after training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from packages.common.configuration import Settings
from packages.models_nn.features.face import FACE_FEATURE_SCHEMA_VERSION

PROTECTED_FACE_METRICS: tuple[str, ...] = ("f1_indoor", "delete_rate_proxy")
LOWER_IS_BETTER: frozenset[str] = frozenset({"delete_rate_proxy", "room_count_error", "false_room_rate"})


@dataclass(frozen=True)
class PromotionDecision:
    accepted: bool
    reasons: tuple[str, ...]
    candidate_metric: float | None
    production_metric: float | None


def promotion_gate(
    *,
    candidate_metrics: Mapping[str, Any],
    production_metrics: Mapping[str, Any] | None,
    feature_schema_version: str,
    artifact_ok: bool,
    held_out_passed: bool | None = None,
    settings: Settings | None = None,
    protected_metrics: Sequence[str] = PROTECTED_FACE_METRICS,
) -> PromotionDecision:
    cfg = settings or Settings()
    reasons: list[str] = []
    flat_candidate = _flatten(candidate_metrics)
    flat_production = _flatten(production_metrics)
    if not artifact_ok:
        reasons.append("artifact is not valid")
    if feature_schema_version != FACE_FEATURE_SCHEMA_VERSION:
        reasons.append(
            f"feature schema {feature_schema_version} is incompatible with {FACE_FEATURE_SCHEMA_VERSION}"
        )
    passed = held_out_passed
    if passed is None:
        passed = bool(flat_candidate.get("held_out_passed", 0.0) >= 1.0)
    if not passed:
        reasons.append("held-out evaluation did not pass")
    target = cfg.face_promote_target_metric
    candidate_value = flat_candidate.get(target)
    production_value = flat_production.get(target)
    if candidate_value is None:
        reasons.append(f"candidate is missing target metric {target}")
    else:
        if candidate_value < cfg.face_promote_min_macro_f1 and target == "macro_f1":
            reasons.append(
                f"{target} {candidate_value:.4f} is below minimum {cfg.face_promote_min_macro_f1:.4f}"
            )
        if production_value is not None and not _beats(candidate_value, production_value, target):
            reasons.append(
                f"{target} {candidate_value:.4f} does not beat production {production_value:.4f}"
            )
    tolerance = cfg.face_promote_tolerance
    for metric in protected_metrics:
        cand = flat_candidate.get(metric)
        prod = flat_production.get(metric)
        if cand is None or prod is None:
            continue
        if _regressed(cand, prod, metric, tolerance):
            reasons.append(
                f"protected metric {metric} regressed beyond tolerance "
                f"(candidate={cand:.4f}, production={prod:.4f}, tol={tolerance:.4f})"
            )
    return PromotionDecision(
        accepted=not reasons,
        reasons=tuple(reasons),
        candidate_metric=candidate_value,
        production_metric=production_value,
    )


def _beats(candidate: float, production: float, metric: str) -> bool:
    if metric in LOWER_IS_BETTER:
        return candidate < production
    return candidate > production


def _regressed(candidate: float, production: float, metric: str, tolerance: float) -> bool:
    if metric in LOWER_IS_BETTER:
        return candidate > production + tolerance
    return candidate < production - tolerance


def _flatten(metrics: Mapping[str, Any] | None) -> dict[str, float]:
    from packages.eval.metrics import flatten_metrics

    return flatten_metrics(metrics)
