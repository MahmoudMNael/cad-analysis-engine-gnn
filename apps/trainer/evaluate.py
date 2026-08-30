"""Held-out evaluation for GNN-C. Compares against the heuristic on the same split."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from packages.common.configuration import Settings, get_settings
from packages.eval.datasets import FaceGraphSample
from packages.eval.metrics import face_classification_metrics, flatten_metrics
from packages.models_nn.common import INDEX_TO_FACE_CLASS
from packages.models_nn.heuristics import HeuristicFaceClassifier
from packages.models_nn.features.face import FACE_NODE_FEATURE_KEYS


def evaluate_graphs(model: Any, graphs: Sequence[FaceGraphSample]) -> dict[str, Any]:
    y_true: list[str] = []
    y_pred: list[str] = []
    for graph in graphs:
        if not graph.face_ids:
            continue
        probas = model.predict_proba(graph.node_features, graph.edge_index, graph.edge_attr)
        for i, label_idx in enumerate(graph.labels):
            if label_idx < 0 or graph.weights[i] <= 0:
                continue
            y_true.append(INDEX_TO_FACE_CLASS[label_idx])
            predicted = max(probas[i], key=probas[i].get)
            y_pred.append(predicted)
    return face_classification_metrics(y_true, y_pred)


def evaluate_heuristic(graphs: Sequence[FaceGraphSample], settings: Settings | None = None) -> dict[str, Any]:
    cfg = settings or Settings()
    clf = HeuristicFaceClassifier()
    y_true: list[str] = []
    y_pred: list[str] = []
    for graph in graphs:
        for i, face_id in enumerate(graph.face_ids):
            if graph.labels[i] < 0 or graph.weights[i] <= 0:
                continue
            features = {FACE_NODE_FEATURE_KEYS[j]: graph.node_features[i][j] for j in range(len(FACE_NODE_FEATURE_KEYS))}
            score = clf.predict(face_id, features, cfg)
            y_true.append(INDEX_TO_FACE_CLASS[graph.labels[i]])
            y_pred.append(score.predicted_class)
    return face_classification_metrics(y_true, y_pred)


def evaluate_face_checkpoint(
    model: Any,
    held_out: Sequence[FaceGraphSample],
    *,
    train_graphs: Sequence[FaceGraphSample] | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    cfg = settings or Settings()
    metrics = evaluate_graphs(model, held_out)
    heuristic = evaluate_heuristic(held_out, cfg)
    metrics["heuristic_macro_f1"] = float(heuristic.get("macro_f1") or 0.0)
    metrics["heuristic_delete_rate_proxy"] = float(heuristic.get("delete_rate_proxy") or 0.0)
    metrics["heuristic_f1_indoor"] = float((heuristic.get("per_class_f1") or {}).get("indoor") or 0.0)
    metrics["held_out_buildings"] = float(len({g.building_id for g in held_out}))
    if train_graphs is not None:
        metrics["train_macro_f1"] = float(evaluate_graphs(model, train_graphs).get("macro_f1") or 0.0)
    return metrics


def main() -> None:
    import argparse
    from pathlib import Path

    import psycopg

    from packages.eval.datasets import face_graphs_from_rows, read_compiled_lake
    from packages.models_nn.face_gnn import FaceGNN
    from packages.models_nn.registry import uri_to_path
    from packages.store.object_store import ObjectStore
    from packages.store.db.repositories.models import ModelRegistryRepository

    parser = argparse.ArgumentParser(description="Evaluate a GNN-C checkpoint on the compiled lake")
    parser.add_argument("--version", required=True)
    parser.add_argument("--lake", default=None)
    args = parser.parse_args()
    settings = get_settings()
    store = ObjectStore(settings.object_store_dir)
    with psycopg.connect(settings.database_url) as conn:
        registry = ModelRegistryRepository(conn)
        row = registry.get("face", args.version)
        if row is None or not row.get("checkpoint_uri"):
            raise SystemExit(f"no checkpoint for face {args.version}")
        model = FaceGNN.load(uri_to_path(str(row["checkpoint_uri"])))
        lake_dir = Path(args.lake) if args.lake else store.root / "lake"
        lake = read_compiled_lake(lake_dir)
        graphs = face_graphs_from_rows(lake.face_cls)
        metrics = evaluate_face_checkpoint(model, graphs, settings=settings)
    print(json.dumps(flatten_metrics(metrics) | {"n": metrics.get("n")}, indent=2))


if __name__ == "__main__":
    main()
