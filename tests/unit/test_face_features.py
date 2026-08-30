from shapely.geometry import Polygon

from packages.assemble.face_features import extract_face_features
from packages.assemble.polygonize import Face, FaceSet
from packages.graphkit.models import PlanGraph


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
    from packages.models_nn.features.face import FACE_NODE_FEATURE_KEYS, vectorize_face_features

    vector = vectorize_face_features(feats)
    assert len(vector) == len(FACE_NODE_FEATURE_KEYS)


def test_adjacent_faces_get_shared_boundary_edge():
    from shapely.geometry import Polygon

    from packages.models_nn.features.face import FACE_EDGE_FEATURE_KEYS, build_face_adjacency

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
