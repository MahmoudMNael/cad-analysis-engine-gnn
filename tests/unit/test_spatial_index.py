from shapely.geometry import LineString, Point, box

from packages.common.spatial_index import SpatialIndex, expand_bounds, query_dwithin, query_ids


def test_expand_bounds_is_aabb_not_buffer():
    line = LineString([(0, 0), (10, 0)])
    probe = expand_bounds(line, 1.0)
    assert probe.geom_type == "Polygon"
    minx, miny, maxx, maxy = probe.bounds
    assert minx == -1.0
    assert miny == -1.0
    assert maxx == 11.0
    assert maxy == 1.0


def test_query_ids_finds_nearby_only():
    index = SpatialIndex.build(
        [
            ("a", Point(0, 0)),
            ("b", Point(10, 0)),
            ("c", Point(0.5, 0.5)),
        ]
    )
    hits = set(query_ids(index, box(-1, -1, 1, 1)))
    assert hits == {"a", "c"}


def test_query_dwithin_points():
    index = SpatialIndex.build(
        [
            ("a", Point(0, 0)),
            ("b", Point(2, 0)),
        ]
    )
    hits = set(query_dwithin(index, Point(0, 0), 1.0))
    assert hits == {"a"}
