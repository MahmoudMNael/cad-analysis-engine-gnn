"""Promote a GNN-C candidate only when the held-out quality gate passes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from packages.common.configuration import Settings, get_settings
from packages.eval.metrics import flatten_metrics
from packages.eval.promotion import promotion_gate
from packages.models_nn.features.face import FACE_FEATURE_SCHEMA_VERSION
from packages.models_nn.registry import is_gnn_ref, uri_to_path
from packages.store.db.repositories.models import ModelRegistryRepository
from packages.store.object_store import ObjectStore


@dataclass
class PromoteResult:
    version: str
    accepted: bool
    reasons: tuple[str, ...]
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "status": self.status,
        }


def production_metrics_for_gate(
    *,
    production: dict[str, Any] | None,
    candidate_metrics: dict[str, Any],
) -> dict[str, float]:
    """Heuristic production is scored on the same held-out set during training."""
    if production and is_gnn_ref(str(production.get("version") or "")):
        return flatten_metrics(production.get("metrics_json") or {})
    return {
        "macro_f1": float(candidate_metrics.get("heuristic_macro_f1") or 0.0),
        "f1_indoor": float(candidate_metrics.get("heuristic_f1_indoor") or 0.0),
        "delete_rate_proxy": float(candidate_metrics.get("heuristic_delete_rate_proxy") or 0.0),
    }


def artifact_is_valid(checkpoint_uri: str | None, object_store: ObjectStore | None = None) -> bool:
    if not checkpoint_uri:
        return False
    path = uri_to_path(checkpoint_uri)
    if path.is_file() and path.stat().st_size > 0:
        return True
    if object_store is None:
        return False
    key = str(path.relative_to(object_store.root)) if path.is_absolute() else str(path)
    try:
        return object_store.exists(key)
    except Exception:
        return False


def promote_face(
    *,
    conn: Any,
    version: str,
    settings: Settings | None = None,
    object_store: ObjectStore | None = None,
) -> PromoteResult:
    cfg = settings or get_settings()
    registry = ModelRegistryRepository(conn)
    candidate = registry.get("face", version)
    if candidate is None:
        return PromoteResult(version=version, accepted=False, reasons=("candidate not found",), status="missing")
    metrics = dict(candidate.get("metrics_json") or {})
    production = registry.get_promoted("face")
    decision = promotion_gate(
        candidate_metrics=metrics,
        production_metrics=production_metrics_for_gate(production=production, candidate_metrics=metrics),
        feature_schema_version=str(candidate.get("feature_schema_version") or FACE_FEATURE_SCHEMA_VERSION),
        artifact_ok=artifact_is_valid(candidate.get("checkpoint_uri"), object_store),
        held_out_passed=bool(float(flatten_metrics(metrics).get("held_out_passed", 0.0)) >= 1.0),
        settings=cfg,
    )
    if decision.accepted:
        registry.promote("face", version)
        return PromoteResult(version=version, accepted=True, reasons=(), status="promoted")
    registry.reject("face", version)
    return PromoteResult(
        version=version,
        accepted=False,
        reasons=decision.reasons,
        status="rejected",
    )


def main() -> None:
    import argparse

    import psycopg

    parser = argparse.ArgumentParser(description="Promote a GNN-C candidate if the held-out gate passes")
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    settings = get_settings()
    store = ObjectStore(settings.object_store_dir)
    with psycopg.connect(settings.database_url) as conn:
        result = promote_face(conn=conn, version=args.version, settings=settings, object_store=store)
    print(json.dumps(result.as_dict(), indent=2))
    if not result.accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
