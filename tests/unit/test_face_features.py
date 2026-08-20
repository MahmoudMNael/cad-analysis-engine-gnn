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
    ):
        assert key in feats
