"""Face-classification and layout-proxy metrics for GNN-C held-out evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from packages.contracts.enums import FaceClass
from packages.models_nn.common import FACE_CLASSES

ROOM_FACE_CLASSES = {FaceClass.INDOOR.value, FaceClass.CORRIDOR.value}


def face_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    classes: Sequence[str] = FACE_CLASSES,
) -> dict[str, Any]:
    labels = list(classes)
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
    macro = float(sum(f1_values) / len(f1_values)) if f1_values else 0.0
    layout = layout_proxy_metrics(y_true, y_pred)
    return {
        "macro_f1": macro,
        "per_class_f1": per_class,
        "confusion_matrix": matrix,
        "n": n,
        **layout,
    }


def layout_proxy_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    gold_rooms = sum(1 for label in y_true if label in ROOM_FACE_CLASSES)
    pred_rooms = sum(1 for label in y_pred if label in ROOM_FACE_CLASSES)
    count_error = abs(pred_rooms - gold_rooms) / max(gold_rooms, 1)
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
    n = max(len(list(y_true)), 1)
    matched = sum(
        1
        for truth, pred in zip(y_true, y_pred, strict=False)
        if (truth in ROOM_FACE_CLASSES) == (pred in ROOM_FACE_CLASSES)
    )
    return {
        "room_count_error": float(count_error),
        "false_room_rate": float(false_rooms / n),
        "delete_rate_proxy": float(deletes / max(gold_rooms, 1)),
        "room_iou_at_0_5": float(matched / n),
    }


def flatten_metrics(metrics: Mapping[str, Any] | None) -> dict[str, float]:
    if not metrics:
        return {}
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            out[key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            out[key] = float(value)
        elif key == "per_class_f1" and isinstance(value, Mapping):
            for class_name, score in value.items():
                if isinstance(score, (int, float)):
                    out[f"f1_{class_name}"] = float(score)
    return out
