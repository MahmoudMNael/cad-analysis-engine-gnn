from packages.common.configuration import Settings
from packages.graphkit.snapping import snap_endpoints


def test_nearby_endpoints_snap_together():
    settings = Settings(snap_tolerance_m=0.02)
    snapped = snap_endpoints([(0.0, 0.0), (0.01, 0.0), (5.0, 0.0)], settings)
    assert snapped[0] == snapped[1]
    assert snapped[2] != snapped[0]


def test_distant_endpoints_do_not_snap():
    settings = Settings(snap_tolerance_m=0.02)
    snapped = snap_endpoints([(0.0, 0.0), (1.0, 0.0)], settings)
    assert snapped[0] != snapped[1]
