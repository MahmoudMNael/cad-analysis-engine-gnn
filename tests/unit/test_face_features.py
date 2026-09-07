from shapely.geometry import LineString, Polygon

from packages.assemble.face_features import build_face_feature_index, extract_face_features
from packages.assemble.polygonize import Face, FaceSet
from packages.graphkit.models import GraphEdge, PlanGraph
from packages.models_nn.features.face import FACE_EDGE_FEATURE_KEYS, FACE_NODE_FEATURE_KEYS, build_face_adjacency, vectorize_face_features


def _face(fid: str, poly: Polygon) -> Face:
    return Face(
        id=fid,
        geometry=poly,
        area=poly.area,
        perimeter=poly.length,
        repaired=False,
        valid=True,
        discarded_reason=None,
    )


def test_face_features_include_required_keys():
    poly = Polygon([(0, 0), (5, 0), (5, 4), (0, 4)])
    face = Face(
        id="f",
        geometry=poly,
        area=poly.area,
        perimeter=poly.length,
        repaired=False,
        valid=True,
        discarded_reason=None,
    )
    faces = FaceSet(faces=(face,), unbounded_edge_ids=())
    graph = PlanGraph(nodes=(), edges=(), portals=())
    feats = extract_face_features(face, faces, graph, (), (), ())
    for key in (
        "area",
        "perimeter",
        "compactness",
        "aspect_ratio",
        "number_of_doors",
        "number_of_windows",
        "wall_overlap",
        "text_overlap",
        "adjacent_face_count",
        "adjacent_unbounded",
        "boundary_complexity",
        "centroid_x",
        "bbox_minx",
    ):
        assert key in feats
    vector = vectorize_face_features(feats)
    assert len(vector) == len(FACE_NODE_FEATURE_KEYS)


def test_adjacent_faces_get_shared_boundary_edge():
    a = Polygon([(0, 0), (5, 0), (5, 4), (0, 4)])
    b = Polygon([(5, 0), (10, 0), (10, 4), (5, 4)])
    edges, attrs = build_face_adjacency(
        [("a", a, a.area, a.length), ("b", b, b.area, b.length)],
    )
    assert (0, 1) in edges
    assert (1, 0) in edges
    assert attrs
    assert len(attrs[0]) == len(FACE_EDGE_FEATURE_KEYS)
    assert attrs[0][0] > 0


def test_three_adjacent_rooms_feature_counts():
    a = Polygon([(0, 0), (5, 0), (5, 4), (0, 4)])
    b = Polygon([(5, 0), (10, 0), (10, 4), (5, 4)])
    c = Polygon([(10, 0), (15, 0), (15, 4), (10, 4)])
    faces = FaceSet(faces=(_face("a", a), _face("b", b), _face("c", c)))
    wall = GraphEdge(
        id="w",
        node_a="n1",
        node_b="n2",
        geometry=LineString([(5, 0), (5, 4)]),
        primitive_ids=(),
        class_prior="wall",
        predicted_class="wall",
        confidence=1.0,
    )
    graph = PlanGraph(nodes=(), edges=(wall,), portals=())
    feats_a = extract_face_features(faces.faces[0], faces, graph, (), (), ())
    feats_b = extract_face_features(faces.faces[1], faces, graph, (), (), ())
    feats_c = extract_face_features(faces.faces[2], faces, graph, (), (), ())
    assert feats_a["adjacent_face_count"] == 1.0
    assert feats_b["adjacent_face_count"] == 2.0
    assert feats_c["adjacent_face_count"] == 1.0


def test_l_shape_shared_boundary_adjacency():
    l_poly = Polygon([(0, 0), (6, 0), (6, 3), (3, 3), (3, 6), (0, 6)])
    square = Polygon([(3, 3), (6, 3), (6, 6), (3, 6)])
    edges, attrs = build_face_adjacency(
        [("l", l_poly, l_poly.area, l_poly.length), ("s", square, square.area, square.length)],
    )
    assert (0, 1) in edges
    assert (1, 0) in edges
    assert attrs[0][0] > 0


def _brute_face_adjacency(faces, portals=(), tolerance=1e-4):
    from packages.models_nn.features.face import shared_boundary_features, vectorize_edge_features

    edge_index = []
    edge_attr = []
    for i, (_, geom_i, area_i, perim_i) in enumerate(faces):
        for j in range(i + 1, len(faces)):
            _, geom_j, area_j, perim_j = faces[j]
            feats = shared_boundary_features(
                geom_i,
                geom_j,
                area_a=area_i,
                area_b=area_j,
                perimeter_a=perim_i,
                perimeter_b=perim_j,
                portals=portals,
                tolerance=tolerance,
            )
            if feats is None:
                continue
            vec = vectorize_edge_features(feats)
            edge_index.append((i, j))
            edge_attr.append(vec)
            edge_index.append((j, i))
            edge_attr.append(vec)
    return edge_index, edge_attr


def test_indexed_face_features_match_brute_four_faces():
    a = Polygon([(0, 0), (5, 0), (5, 4), (0, 4)])
    b = Polygon([(5, 0), (10, 0), (10, 4), (5, 4)])
    c = Polygon([(0, 4), (5, 4), (5, 8), (0, 8)])
    d = Polygon([(5, 4), (10, 4), (10, 8), (5, 8)])
    faces = FaceSet(faces=(_face("a", a), _face("b", b), _face("c", c), _face("d", d)))
    graph = PlanGraph(nodes=(), edges=(), portals=())
    index = build_face_feature_index(faces, graph, (), ())
    for face in faces.faces:
        brute = extract_face_features(face, faces, graph, (), (), (), index=None)
        indexed = extract_face_features(face, faces, graph, (), (), (), index=index)
        assert brute == indexed


def test_five_room_grid_adjacency_matches_brute():
    polys = []
    for row in range(5):
        polys.append(Polygon([(row * 4, 0), (row * 4 + 4, 0), (row * 4 + 4, 3), (row * 4, 3)]))
    payload = [(f"f{i}", p, p.area, p.length) for i, p in enumerate(polys)]
    assert build_face_adjacency(payload) == _brute_face_adjacency(payload)


def test_envelope_with_interior_faces_matches_brute():
    outer = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    inners = [
        Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)])
        for x, y in ((2, 2), (5, 5), (8, 8), (11, 2), (14, 14))
    ]
    payload = [("outer", outer, outer.area, outer.length)] + [
        (f"in{i}", p, p.area, p.length) for i, p in enumerate(inners)
    ]
    indexed = build_face_adjacency(payload)
    brute = _brute_face_adjacency(payload)
    assert indexed == brute
    assert indexed == ([], [])


def test_hole_ring_shared_boundary_matches_brute():
    hole = [(3, 3), (3, 7), (7, 7), (7, 3), (3, 3)]
    outer = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)], holes=[hole])
    inner = Polygon([(3, 3), (7, 3), (7, 7), (3, 7), (3, 3)])
    payload = [
        ("outer", outer, outer.area, outer.length),
        ("inner", inner, inner.area, inner.length),
    ]
    indexed = build_face_adjacency(payload)
    brute = _brute_face_adjacency(payload)
    assert indexed == brute
    assert (0, 1) in indexed[0]
    assert (1, 0) in indexed[0]
    assert indexed[1][0][0] > 0


def test_portal_on_shared_boundary_matches_brute():
    a = Polygon([(0, 0), (5, 0), (5, 4), (0, 4)])
    b = Polygon([(5, 0), (10, 0), (10, 4), (5, 4)])
    portals = (LineString([(5, 1), (5, 3)]),)
    payload = [("a", a, a.area, a.length), ("b", b, b.area, b.length)]
    indexed = build_face_adjacency(payload, portals)
    brute = _brute_face_adjacency(payload, portals)
    assert indexed == brute
    assert indexed[1]
    portal_idx = FACE_EDGE_FEATURE_KEYS.index("portal_on_boundary")
    assert indexed[1][0][portal_idx] == 1.0

