from shapely.geometry import LineString

from packages.common.configuration import Settings
from packages.common.spatial_index import SpatialIndex
from packages.contracts.enums import PrimitiveKind
from packages.graphkit.models import GraphEdge
from packages.graphkit.portals import project_opening_onto_walls
from packages.ingest.models import primitive_from_parts


def _edge(eid: str, a: tuple[float, float], b: tuple[float, float]) -> GraphEdge:
    return GraphEdge(
        id=eid,
        node_a="n1",
        node_b="n2",
        geometry=LineString([a, b]),
        primitive_ids=(),
        class_prior="wall",
        predicted_class="wall",
        confidence=1.0,
    )


def test_door_swing_projects_onto_wall_gap():
    wall = _edge("e1", (0.0, 2.5), (0.0, 4.0))
    edge_by_id = {wall.id: wall}
    index = SpatialIndex.build([(wall.id, wall.geometry)])
    prim = primitive_from_parts(
        primitive_id="door",
        kind=PrimitiveKind.ARC,
        geometry=LineString([(0.9, 1.6), (0.64, 2.24), (0.0, 2.5)]),
        layer="DOOR",
        color=7,
        linetype="CONTINUOUS",
        source_entity_type="ARC",
        source_handle="d1",
        block_name="DOOR",
        attributes={"center": (0.0, 1.6), "radius": 0.9},
    )
    line = project_opening_onto_walls(prim, index, edge_by_id, Settings())
    assert line is not None
    xs = [line.coords[0][0], line.coords[-1][0]]
    ys = sorted([line.coords[0][1], line.coords[-1][1]])
    assert max(xs) <= 0.05
    assert ys[0] <= 1.65
    assert ys[1] >= 2.45


def test_interior_curve_without_nearby_wall_is_dropped():
    wall = _edge("e1", (0.0, 0.0), (5.0, 0.0))
    edge_by_id = {wall.id: wall}
    index = SpatialIndex.build([(wall.id, wall.geometry)])
    prim = primitive_from_parts(
        primitive_id="swing",
        kind=PrimitiveKind.ARC,
        geometry=LineString([(2.0, 2.0), (2.4, 2.4), (2.0, 2.8)]),
        layer="DOOR",
        color=7,
        linetype="CONTINUOUS",
        source_entity_type="ARC",
        source_handle="d2",
        block_name="DOOR",
        attributes={"center": (2.0, 2.4), "radius": 0.4},
    )
    assert project_opening_onto_walls(prim, index, edge_by_id, Settings()) is None
