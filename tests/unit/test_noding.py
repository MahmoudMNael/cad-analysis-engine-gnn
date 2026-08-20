from packages.common.configuration import Settings
from packages.graphkit.noding import RawSegment, node_segments


def test_crossing_segments_are_split():
    settings = Settings()
    segs = [
        RawSegment("a", (0.0, 1.0), (2.0, 1.0)),
        RawSegment("b", (1.0, 0.0), (1.0, 2.0)),
    ]
    noded = node_segments(segs, settings)
    assert len(noded) == 4
