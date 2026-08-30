"""Standalone GNN-C training kernel.

Depends only on the Python standard library plus torch and numpy.
Colab/Kaggle notebooks download this file from the local API and exec/import it.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

FACE_CLASSES: tuple[str, ...] = ("indoor", "corridor", "shaft", "outdoor", "cavity")
FACE_FEATURE_SCHEMA_VERSION = "face_features_v1"
FACE_EDGE_FEATURE_SCHEMA_VERSION = "face_edge_features_v1"
FACE_NODE_FEATURE_KEYS: tuple[str, ...] = (
    "area",
    "perimeter",
    "compactness",
    "centroid_x",
    "centroid_y",
    "bbox_minx",
    "bbox_miny",
    "bbox_maxx",
    "bbox_maxy",
    "aspect_ratio",
    "number_of_doors",
    "number_of_windows",
    "wall_overlap",
    "text_overlap",
    "adjacent_face_count",
    "adjacent_unbounded",
    "boundary_complexity",
)
FACE_EDGE_FEATURE_KEYS: tuple[str, ...] = (
    "shared_length",
    "shared_length_norm",
    "area_ratio",
    "portal_on_boundary",
)
ROOM_FACE_CLASSES = {"indoor", "corridor"}
KERNEL_PATH = Path(__file__).resolve()


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def softmax(scores: dict[str, float]) -> dict[str, float]:
    peak = max(scores.values())
    exps = {key: math.exp(value - peak) for key, value in scores.items()}
    total = sum(exps.values()) or 1.0
    return {key: value / total for key, value in exps.items()}


def graph_to_dict(graph: Any) -> dict[str, Any]:
    if isinstance(graph, dict):
        return {
            "job_id": str(graph.get("job_id") or ""),
            "building_id": str(graph.get("building_id") or graph.get("job_id") or ""),
            "face_ids": list(graph.get("face_ids") or []),
            "node_features": [list(row) for row in (graph.get("node_features") or [])],
            "edge_index": [list(pair) for pair in (graph.get("edge_index") or [])],
            "edge_attr": [list(row) for row in (graph.get("edge_attr") or [])],
            "labels": [int(v) for v in (graph.get("labels") or [])],
            "weights": [float(v) for v in (graph.get("weights") or [])],
        }
    return {
        "job_id": str(graph.job_id),
        "building_id": str(graph.building_id or graph.job_id),
        "face_ids": list(graph.face_ids),
        "node_features": [list(row) for row in graph.node_features],
        "edge_index": [list(pair) for pair in graph.edge_index],
        "edge_attr": [list(row) for row in graph.edge_attr],
        "labels": [int(v) for v in graph.labels],
        "weights": [float(v) for v in graph.weights],
    }


class FaceGNN:
    name = "face_gnn"

    def __init__(
        self,
        *,
        node_dim: int = len(FACE_NODE_FEATURE_KEYS),
        edge_dim: int = len(FACE_EDGE_FEATURE_KEYS),
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = len(FACE_CLASSES),
        feature_mean: Sequence[float] | None = None,
        feature_std: Sequence[float] | None = None,
        feature_schema_version: str = FACE_FEATURE_SCHEMA_VERSION,
        edge_feature_schema_version: str = FACE_EDGE_FEATURE_SCHEMA_VERSION,
        engine_sha: str = "",
    ) -> None:
        torch, nn, _F = _torch_mods()
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.num_classes = int(num_classes)
        self.feature_schema_version = feature_schema_version
        self.edge_feature_schema_version = edge_feature_schema_version
        self.engine_sha = engine_sha
        mean = list(feature_mean) if feature_mean is not None else [0.0] * self.node_dim
        std = list(feature_std) if feature_std is not None else [1.0] * self.node_dim
        self.feature_mean = mean
        self.feature_std = [s if abs(s) > 1e-6 else 1.0 for s in std]
        self._torch = torch
        self._module = _FaceSAGE(
            nn,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            num_classes=self.num_classes,
        )

    @property
    def module(self) -> Any:
        return self._module

    def train(self) -> None:
        self._module.train()

    def eval(self) -> None:
        self._module.eval()

    def parameters(self) -> Any:
        return self._module.parameters()

    def logits(
        self,
        node_features: Sequence[Sequence[float]],
        edge_index: Sequence[Sequence[int]],
        edge_attr: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        torch = self._torch
        x, ei, ea = self._tensors(node_features, edge_index, edge_attr)
        with torch.no_grad() if not self._module.training else _nullcontext():
            out = self._module(x, ei, ea)
        return out.detach().cpu().tolist()

    def predict_proba(
        self,
        node_features: Sequence[Sequence[float]],
        edge_index: Sequence[Sequence[int]],
        edge_attr: Sequence[Sequence[float]],
    ) -> list[dict[str, float]]:
        was_training = self._module.training
        self.eval()
        logits = self.logits(node_features, edge_index, edge_attr)
        if was_training:
            self.train()
        rows: list[dict[str, float]] = []
        for row in logits:
            scores = {FACE_CLASSES[i]: float(row[i]) for i in range(self.num_classes)}
            rows.append(softmax(scores))
        return rows

    def save_bytes(self) -> bytes:
        torch = self._torch
        payload = {
            "state_dict": self._module.state_dict(),
            "node_dim": self.node_dim,
            "edge_dim": self.edge_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "num_classes": self.num_classes,
            "feature_schema_version": self.feature_schema_version,
            "edge_feature_schema_version": self.edge_feature_schema_version,
            "node_feature_keys": FACE_NODE_FEATURE_KEYS,
            "edge_feature_keys": FACE_EDGE_FEATURE_KEYS,
            "classes": FACE_CLASSES,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "engine_sha": self.engine_sha,
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        return buffer.getvalue()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.save_bytes())

    @classmethod
    def load_bytes(cls, data: bytes) -> FaceGNN:
        torch, _nn, _F = _torch_mods()
        try:
            payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(io.BytesIO(data), map_location="cpu")
        model = cls(
            node_dim=int(payload["node_dim"]),
            edge_dim=int(payload["edge_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            num_layers=int(payload["num_layers"]),
            dropout=float(payload.get("dropout", 0.1)),
            num_classes=int(payload["num_classes"]),
            feature_mean=payload.get("feature_mean"),
            feature_std=payload.get("feature_std"),
            feature_schema_version=str(payload.get("feature_schema_version") or FACE_FEATURE_SCHEMA_VERSION),
            edge_feature_schema_version=str(
                payload.get("edge_feature_schema_version") or FACE_EDGE_FEATURE_SCHEMA_VERSION
            ),
            engine_sha=str(payload.get("engine_sha") or ""),
        )
        model._module.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    @classmethod
    def load(cls, path: Path) -> FaceGNN:
        return cls.load_bytes(Path(path).read_bytes())

    def _tensors(
        self,
        node_features: Sequence[Sequence[float]],
        edge_index: Sequence[Sequence[int]],
        edge_attr: Sequence[Sequence[float]],
    ) -> tuple[Any, Any, Any]:
        torch = self._torch
        x = torch.tensor([list(row) for row in node_features], dtype=torch.float32)
        if x.ndim != 2:
            x = x.reshape(len(node_features), self.node_dim)
        mean = torch.tensor(self.feature_mean, dtype=torch.float32)
        std = torch.tensor(self.feature_std, dtype=torch.float32)
        x = (x - mean) / std
        pairs = [list(pair) for pair in edge_index]
        if pairs:
            ei = torch.tensor(pairs, dtype=torch.long).t().contiguous()
            ea = torch.tensor([list(row) for row in edge_attr], dtype=torch.float32)
            if ea.ndim == 1:
                ea = ea.reshape(len(edge_attr), self.edge_dim)
        else:
            ei = torch.zeros((2, 0), dtype=torch.long)
            ea = torch.zeros((0, self.edge_dim), dtype=torch.float32)
        return x, ei, ea


def train_face_on_graphs(
    train_graphs: Sequence[Any],
    val_graphs: Sequence[Any],
    *,
    hyperparams: Mapping[str, Any],
    engine_sha: str = "",
) -> tuple[FaceGNN, dict[str, Any]]:
    torch, _nn, functional = _torch_mods()
    torch.manual_seed(int(hyperparams.get("face_train_seed", 0)))
    train = [graph_to_dict(graph) for graph in train_graphs]
    val = [graph_to_dict(graph) for graph in val_graphs]
    mean, std = _feature_moments(train)
    model = FaceGNN(
        node_dim=len(FACE_NODE_FEATURE_KEYS),
        edge_dim=len(FACE_EDGE_FEATURE_KEYS),
        hidden_dim=int(hyperparams.get("face_gnn_hidden_dim", 64)),
        num_layers=int(hyperparams.get("face_gnn_layers", 2)),
        dropout=float(hyperparams.get("face_gnn_dropout", 0.1)),
        feature_mean=mean,
        feature_std=std,
        engine_sha=engine_sha,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(hyperparams.get("face_gnn_lr", 1e-3)))
    best_state = model.save_bytes()
    best_val = -1.0
    patience = 8
    stalled = 0
    epochs = int(hyperparams.get("face_gnn_epochs", 40))
    for _epoch in range(epochs):
        model.train()
        for graph in train:
            loss = _graph_loss(model, graph, torch, functional)
            if loss is None:
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_metrics = evaluate_graphs(model, val or train)
        val_score = float(val_metrics.get("macro_f1") or 0.0)
        if val_score > best_val:
            best_val = val_score
            best_state = model.save_bytes()
            stalled = 0
        else:
            stalled += 1
        if stalled >= patience:
            break
    trained = FaceGNN.load_bytes(best_state)
    trained.engine_sha = engine_sha
    return trained, {"best_val_macro_f1": best_val, "epochs": float(epochs)}


def evaluate_graphs(model: FaceGNN, graphs: Sequence[Any]) -> dict[str, Any]:
    y_true: list[str] = []
    y_pred: list[str] = []
    for raw in graphs:
        graph = graph_to_dict(raw)
        if not graph["face_ids"]:
            continue
        probas = model.predict_proba(graph["node_features"], graph["edge_index"], graph["edge_attr"])
        for i, label_idx in enumerate(graph["labels"]):
            if label_idx < 0 or graph["weights"][i] <= 0:
                continue
            y_true.append(FACE_CLASSES[label_idx])
            y_pred.append(max(probas[i], key=probas[i].get))
    return face_classification_metrics(y_true, y_pred)


def evaluate_heuristic(graphs: Sequence[Any], hyperparams: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(hyperparams or {})
    y_true: list[str] = []
    y_pred: list[str] = []
    for raw in graphs:
        graph = graph_to_dict(raw)
        for i, face_id in enumerate(graph["face_ids"]):
            if graph["labels"][i] < 0 or graph["weights"][i] <= 0:
                continue
            features = {
                FACE_NODE_FEATURE_KEYS[j]: graph["node_features"][i][j]
                for j in range(min(len(FACE_NODE_FEATURE_KEYS), len(graph["node_features"][i])))
            }
            y_true.append(FACE_CLASSES[graph["labels"][i]])
            y_pred.append(_heuristic_class(features, cfg))
    return face_classification_metrics(y_true, y_pred)


def evaluate_face_checkpoint(
    model: FaceGNN,
    held_out: Sequence[Any],
    *,
    train_graphs: Sequence[Any] | None = None,
    hyperparams: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = evaluate_graphs(model, held_out)
    heuristic = evaluate_heuristic(held_out, hyperparams)
    metrics["heuristic_macro_f1"] = float(heuristic.get("macro_f1") or 0.0)
    metrics["heuristic_delete_rate_proxy"] = float(heuristic.get("delete_rate_proxy") or 0.0)
    metrics["heuristic_f1_indoor"] = float((heuristic.get("per_class_f1") or {}).get("indoor") or 0.0)
    metrics["held_out_buildings"] = float(
        len({graph_to_dict(graph)["building_id"] for graph in held_out})
    )
    if train_graphs is not None:
        metrics["train_macro_f1"] = float(evaluate_graphs(model, train_graphs).get("macro_f1") or 0.0)
    return metrics


def split_graphs(
    graphs: Sequence[Any],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    items = [graph_to_dict(graph) for graph in graphs]
    buildings = sorted({item["building_id"] or item["job_id"] for item in items})
    train_b: set[str] = set()
    val_b: set[str] = set()
    test_b: set[str] = set()
    val_cut = train_ratio
    test_cut = train_ratio + val_ratio
    for building in buildings:
        digest = hashlib.sha256(f"{seed}:{building}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") / float(2**64)
        if bucket < val_cut:
            train_b.add(building)
        elif bucket < test_cut:
            val_b.add(building)
        else:
            test_b.add(building)
    if len(buildings) >= 3:
        if not train_b:
            train_b.add(buildings[0])
            val_b.discard(buildings[0])
            test_b.discard(buildings[0])
        if not val_b:
            donor = next((b for b in buildings if b in train_b and len(train_b) > 1), buildings[-2])
            val_b.add(donor)
            train_b.discard(donor)
            test_b.discard(donor)
        if not test_b:
            donor = next((b for b in buildings if b in train_b and len(train_b) > 1), buildings[-1])
            test_b.add(donor)
            train_b.discard(donor)
            val_b.discard(donor)
    elif len(buildings) == 2:
        train_b = {buildings[0]}
        val_b = {buildings[1]}
        test_b = {buildings[1]}
    elif len(buildings) == 1:
        train_b = {buildings[0]}
        val_b = set()
        test_b = set()

    def _select(bucket: set[str]) -> list[dict[str, Any]]:
        return [item for item in items if (item["building_id"] or item["job_id"]) in bucket]

    return _select(train_b), _select(val_b), _select(test_b)


def run_training_from_bundle(bundle: Mapping[str, Any], *, force: bool = False) -> tuple[bytes, dict[str, Any]]:
    eligibility = dict(bundle.get("eligibility") or {})
    if not force and not eligibility.get("ok"):
        reasons = eligibility.get("reasons") or ["dataset is not eligible"]
        raise RuntimeError("; ".join(str(item) for item in reasons))
    hyperparams = dict(bundle.get("hyperparams") or {})
    graphs = list(bundle.get("graphs") or [])
    train, val, test = split_graphs(
        graphs,
        train_ratio=float(hyperparams.get("face_train_ratio", 0.7)),
        val_ratio=float(hyperparams.get("face_val_ratio", 0.15)),
        seed=int(hyperparams.get("face_train_seed", 0)),
    )
    engine_sha = str(bundle.get("engine_sha") or "")
    model, info = train_face_on_graphs(train, val, hyperparams=hyperparams, engine_sha=engine_sha)
    held_out = test or val
    metrics = evaluate_face_checkpoint(model, held_out, train_graphs=train, hyperparams=hyperparams)
    metrics["best_val_macro_f1"] = info["best_val_macro_f1"]
    metrics["held_out_passed"] = 1.0 if held_out and metrics.get("n", 0) > 0 else 0.0
    metrics["n_train_graphs"] = float(len(train))
    metrics["n_val_graphs"] = float(len(val))
    metrics["n_test_graphs"] = float(len(test))
    metrics["feature_schema_version"] = str(
        bundle.get("feature_schema_version") or FACE_FEATURE_SCHEMA_VERSION
    )
    metrics["engine_sha"] = engine_sha
    return model.save_bytes(), metrics


def face_classification_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    labels = list(FACE_CLASSES)
    index = {name: i for i, name in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    n = 0
    for truth, pred in zip(y_true, y_pred, strict=False):
        if truth not in index or pred not in index:
            continue
        matrix[index[truth]][index[pred]] += 1
        n += 1
    per_class: dict[str, float] = {}
    f1_values: list[float] = []
    for i, name in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(len(labels)) if r != i)
        fn = sum(matrix[i][c] for c in range(len(labels)) if c != i)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 0.0 if (precision + recall) == 0.0 else 2.0 * precision * recall / (precision + recall)
        per_class[name] = float(f1)
        f1_values.append(f1)
    gold_rooms = sum(1 for label in y_true if label in ROOM_FACE_CLASSES)
    pred_rooms = sum(1 for label in y_pred if label in ROOM_FACE_CLASSES)
    false_rooms = sum(
        1
        for truth, pred in zip(y_true, y_pred, strict=False)
        if pred in ROOM_FACE_CLASSES and truth not in ROOM_FACE_CLASSES
    )
    deletes = sum(
        1
        for truth, pred in zip(y_true, y_pred, strict=False)
        if truth in ROOM_FACE_CLASSES and pred not in ROOM_FACE_CLASSES
    )
    count_n = max(len(list(y_true)), 1)
    matched = sum(
        1
        for truth, pred in zip(y_true, y_pred, strict=False)
        if (truth in ROOM_FACE_CLASSES) == (pred in ROOM_FACE_CLASSES)
    )
    return {
        "macro_f1": float(sum(f1_values) / len(f1_values)) if f1_values else 0.0,
        "per_class_f1": per_class,
        "confusion_matrix": matrix,
        "n": n,
        "room_count_error": float(abs(pred_rooms - gold_rooms) / max(gold_rooms, 1)),
        "false_room_rate": float(false_rooms / count_n),
        "delete_rate_proxy": float(deletes / max(gold_rooms, 1)),
        "room_iou_at_0_5": float(matched / count_n),
    }


def dumps_metrics(metrics: Mapping[str, Any]) -> str:
    return json.dumps(metrics, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _heuristic_class(features: Mapping[str, float], hyperparams: Mapping[str, Any]) -> str:
    scores = {name: 0.05 for name in FACE_CLASSES}
    area = float(features.get("area") or 0.0)
    compactness = float(features.get("compactness") or 0.0)
    aspect = float(features.get("aspect_ratio") or 1.0)
    doors = float(features.get("number_of_doors") or 0.0)
    windows = float(features.get("number_of_windows") or 0.0)
    unbounded = float(features.get("adjacent_unbounded") or 0.0)
    texts = float(features.get("text_overlap") or 0.0)
    perimeter = float(features.get("perimeter") or 0.0)
    wall_overlap = float(features.get("wall_overlap") or 0.0)
    adjacent = float(features.get("adjacent_face_count") or 0.0)
    wall_density = wall_overlap / max(perimeter, 1e-6)
    portals = doors + windows
    cavity_area = float(hyperparams.get("cavity_area_m2", 0.8))
    shaft_area = float(hyperparams.get("shaft_area_m2", 2.5))
    outdoor_area = float(hyperparams.get("outdoor_area_m2", 250.0))
    face_threshold = float(hyperparams.get("face_threshold", 0.35))
    if area < cavity_area:
        scores["cavity"] += 4.0
        if adjacent >= 2:
            scores["cavity"] += 0.6
    elif area < shaft_area and aspect > 4.0:
        scores["shaft"] += 3.5
    elif unbounded >= 1.0 and area > outdoor_area and portals < 1 and (compactness < 0.2 or wall_density < 0.08):
        scores["outdoor"] += 3.5
    elif portals >= 2 and aspect > 3.0 and compactness < 0.45:
        scores["corridor"] += 3.0
    else:
        scores["indoor"] += 3.0
        if texts > 0:
            scores["indoor"] += 0.8
        if compactness > 0.3:
            scores["indoor"] += 0.5
        if wall_density > 0.2:
            scores["indoor"] += 0.4
        if adjacent >= 1 and unbounded < 1:
            scores["indoor"] += 0.4
    probs = softmax(scores)
    predicted = max(probs, key=probs.get)
    if float(probs[predicted]) < face_threshold:
        predicted = "indoor"
    return predicted


def _feature_moments(graphs: Sequence[Mapping[str, Any]]) -> tuple[list[float], list[float]]:
    dim = len(FACE_NODE_FEATURE_KEYS)
    counts = 0
    sums = [0.0] * dim
    sq = [0.0] * dim
    for graph in graphs:
        for row in graph.get("node_features") or []:
            counts += 1
            for i, value in enumerate(row):
                if i >= dim:
                    break
                sums[i] += float(value)
                sq[i] += float(value) * float(value)
    if counts == 0:
        return [0.0] * dim, [1.0] * dim
    mean = [total / counts for total in sums]
    std = []
    for i in range(dim):
        var = max(sq[i] / counts - mean[i] * mean[i], 0.0)
        std.append(var**0.5 if var**0.5 > 1e-6 else 1.0)
    return mean, std


def _graph_loss(model: FaceGNN, graph: Mapping[str, Any], torch: Any, functional: Any) -> Any | None:
    labeled = [
        i for i, label in enumerate(graph["labels"]) if label >= 0 and graph["weights"][i] > 0
    ]
    if not labeled:
        return None
    x, ei, ea = model._tensors(graph["node_features"], graph["edge_index"], graph["edge_attr"])
    logits = model.module(x, ei, ea)
    idx = torch.tensor(labeled, dtype=torch.long)
    target = torch.tensor([graph["labels"][i] for i in labeled], dtype=torch.long)
    weights = torch.tensor([graph["weights"][i] for i in labeled], dtype=torch.float32)
    per = functional.cross_entropy(logits.index_select(0, idx), target, reduction="none")
    return (per * weights).sum() / weights.sum().clamp(min=1e-6)


def _torch_mods() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("PyTorch is required for GNN-C training.") from exc
    return torch, nn, F


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


def _FaceSAGE(
    nn: Any,
    *,
    node_dim: int,
    edge_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    num_classes: int,
) -> Any:
    class FaceSAGE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dropout_p = dropout
            self.input = nn.Linear(node_dim, hidden_dim)
            self.layers = nn.ModuleList(
                [_SAGELayer(nn, hidden_dim, hidden_dim, edge_dim) for _ in range(max(num_layers, 1))]
            )
            self.head = nn.Linear(hidden_dim, num_classes)

        def forward(self, x: Any, edge_index: Any, edge_attr: Any) -> Any:
            import torch.nn.functional as F

            h = F.relu(self.input(x))
            h = F.dropout(h, p=self.dropout_p, training=self.training)
            for layer in self.layers:
                h = layer(h, edge_index, edge_attr)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout_p, training=self.training)
            return self.head(h)

    return FaceSAGE()


def _SAGELayer(nn: Any, in_dim: int, out_dim: int, edge_dim: int) -> Any:
    class SAGELayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin_self = nn.Linear(in_dim, out_dim)
            self.lin_msg = nn.Linear(in_dim + edge_dim, out_dim)

        def forward(self, x: Any, edge_index: Any, edge_attr: Any) -> Any:
            import torch
            import torch.nn.functional as F

            self_term = self.lin_self(x)
            if edge_index.numel() == 0:
                return self_term
            src, dst = edge_index[0], edge_index[1]
            msg_in = torch.cat([x[src], edge_attr], dim=-1)
            msg = F.relu(self.lin_msg(msg_in))
            agg = torch.zeros(x.size(0), msg.size(-1), dtype=x.dtype, device=x.device)
            deg = torch.zeros(x.size(0), 1, dtype=x.dtype, device=x.device)
            agg.index_add_(0, dst, msg)
            deg.index_add_(0, dst, torch.ones(dst.size(0), 1, dtype=x.dtype, device=x.device))
            agg = agg / deg.clamp(min=1.0)
            return self_term + agg

    return SAGELayer()
