"""Placeholder for P4. P1 never imports torch on the heuristic path."""


class GapGNN:
    name = "gap_gnn"

    def predict(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError("Gap GNN is not available in P1; use heuristic runtime")
