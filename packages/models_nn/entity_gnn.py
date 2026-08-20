"""Placeholder for P3+. P1 never imports torch on the heuristic path."""


class EntityGNN:
    name = "entity_gnn"

    def predict(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError("Entity GNN is not available in P1; use heuristic runtime")
