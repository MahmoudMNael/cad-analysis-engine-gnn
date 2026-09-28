"""GNN-C FaceClass model. Implementation lives in the portable training kernel.

Torch is imported lazily so the heuristic runtime path never requires it.
When torch-geometric is installed, graphs are packed as PyG `Data` objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from packages.models_nn.face_train_kernel import FaceGNN, torch_available
from packages.models_nn.features.face import FACE_EDGE_FEATURE_KEYS

__all__ = [
    "FaceGNN",
    "pack_pyg_data",
    "pyg_available",
    "torch_available",
]


def pyg_available() -> bool:
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        return False
    return True


def pack_pyg_data(
    node_features: Sequence[Sequence[float]],
    edge_index: Sequence[tuple[int, int]],
    edge_attr: Sequence[Sequence[float]],
    y: Sequence[int] | None = None,
) -> Any:
    """Pack a face graph as torch_geometric.data.Data when PyG is installed."""
    if not pyg_available():
        return {
            "x": [list(row) for row in node_features],
            "edge_index": list(edge_index),
            "edge_attr": [list(row) for row in edge_attr],
            "y": None if y is None else list(y),
        }
    from torch_geometric.data import Data
    import torch

    x = torch.tensor([list(row) for row in node_features], dtype=torch.float32)
    if edge_index:
        ei = torch.tensor([[s, d] for s, d in edge_index], dtype=torch.long).t().contiguous()
        ea = torch.tensor([list(row) for row in edge_attr], dtype=torch.float32)
    else:
        ei = torch.zeros((2, 0), dtype=torch.long)
        ea = torch.zeros((0, len(FACE_EDGE_FEATURE_KEYS)), dtype=torch.float32)
    data = Data(x=x, edge_index=ei, edge_attr=ea)
    if y is not None:
        data.y = torch.tensor(list(y), dtype=torch.long)
    return data
