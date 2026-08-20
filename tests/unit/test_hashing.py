from packages.common.geometry import canonicalize_linestring_coords, canonicalize_polygon_coords
from packages.common.hashing import stable_hash


def test_line_hash_is_direction_invariant():
    a = stable_hash("line", canonicalize_linestring_coords([(0, 0), (1, 0)], 6), {"layer": "WALL"})
    b = stable_hash("line", canonicalize_linestring_coords([(1, 0), (0, 0)], 6), {"layer": "WALL"})
    assert a == b


def test_polygon_hash_is_winding_and_start_invariant():
    ring = [(0, 0), (1, 0), (1, 1), (0, 1)]
    rev = list(reversed(ring))
    shifted = ring[2:] + ring[:2]
    h1 = stable_hash("face", canonicalize_polygon_coords(ring, 6), {})
    h2 = stable_hash("face", canonicalize_polygon_coords(rev, 6), {})
    h3 = stable_hash("face", canonicalize_polygon_coords(shifted, 6), {})
    assert h1 == h2 == h3


def test_unordered_endpoint_pairs_share_hash():
    from packages.common.geometry import canonicalize_unordered_pair

    a = stable_hash("edge", canonicalize_unordered_pair((0, 0), (2, 0), 6), {})
    b = stable_hash("edge", canonicalize_unordered_pair((2, 0), (0, 0), 6), {})
    assert a == b
