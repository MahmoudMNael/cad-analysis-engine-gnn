"""Graph validation after construction and closure."""

from __future__ import annotations

from packages.graphkit.models import PlanGraph


def validate_plan_graph(graph: PlanGraph) -> list[str]:
    errors: list[str] = []
    nodes = graph.node_map()
    edges = graph.edge_map()
    if len(nodes) != len(graph.nodes):
        errors.append("duplicate node ids")
    if len(edges) != len(graph.edges):
        errors.append("duplicate edge ids")
    incident: dict[str, list[str]] = {n.id: [] for n in graph.nodes}
    for edge in graph.edges:
        if edge.node_a not in nodes or edge.node_b not in nodes:
            errors.append(f"edge {edge.id} references missing node")
            continue
        if edge.node_a == edge.node_b:
            errors.append(f"edge {edge.id} is a loop")
        incident.setdefault(edge.node_a, []).append(edge.id)
        incident.setdefault(edge.node_b, []).append(edge.id)
        if edge.geometry.is_empty:
            errors.append(f"edge {edge.id} has empty geometry")
    for node in graph.nodes:
        expected = tuple(sorted(incident.get(node.id, [])))
        actual = tuple(sorted(node.incident_edge_ids))
        if expected != actual:
            errors.append(f"node {node.id} incident edges mismatch")
    for portal in graph.portals:
        for edge_id in portal.source_edge_ids:
            if edge_id not in edges:
                errors.append(f"portal {portal.id} references missing edge {edge_id}")
    return errors
