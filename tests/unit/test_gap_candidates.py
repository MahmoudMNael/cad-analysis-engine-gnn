from shapely.geometry import LineString

from packages.common.configuration import Settings
from packages.graphkit.candidates import generate_gap_candidates
from packages.graphkit.models import GraphEdge, GraphNode, PlanGraph
from packages.common.hashing import stable_hash


def _node(i: str, x: float, y: float, edges: tuple[str, ...]) -> GraphNode:
    return GraphNode(id=i, point=(x, y), incident_edge_ids=edges, source_primitive_ids=())


def test_gap_candidates_from_dangling_nodes_only():
    settings = Settings(gap_max_distance_m=2.0)
    e1 = GraphEdge(
        id="e1",
        node_a="n1",
        node_b="n2",
        geometry=LineString([(0, 0), (1, 0)]),
        primitive_ids=(),
        class_prior="wall",
        predicted_class="wall",
        confidence=1.0,
    )
    graph = PlanGraph(
        nodes=(
            _node("n1", 0, 0, ("e1",)),
            _node("n2", 1, 0, ("e1",)),
            _node("n3", 1.5, 0, ()),
        ),
        edges=(e1,),
        portals=(),
    )
    cands = generate_gap_candidates(graph, settings)
    assert cands
    assert all(c.distance_m <= 2.0 for c in cands)
