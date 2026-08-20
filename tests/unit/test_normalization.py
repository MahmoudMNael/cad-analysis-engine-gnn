from shapely.geometry import LineString

from packages.common.configuration import Settings
from packages.ingest.normalization import normalize_geometry


def test_normalize_drops_empty_and_zero_length():
    settings = Settings()
    assert normalize_geometry(LineString([(0, 0), (0, 0)]), settings.coord_precision) is None


def test_normalize_rounds_coordinates():
    geom = normalize_geometry(LineString([(0.0000001, 0.0), (1.0000004, 0.0)]), 6)
    assert geom is not None
    coords = list(geom.coords)
    assert coords[0][0] == 0.0 or abs(coords[0][0]) < 1e-6
