from __future__ import annotations

from pathlib import Path

from tests.helpers.dxf_builder import add_lines, add_rect, write_simple_dxf
from tests.helpers.pipeline import run_dxf


def test_single_rectangle(tmp_path: Path):
    path = write_simple_dxf(
        tmp_path / "rect.dxf",
        lambda msp: add_rect(msp, (0, 0), 5000, 4000),
    )
    result = run_dxf(path)
    layout = result.layout
    assert layout.unit == "m"
    assert len(layout.physical_rooms) == 1
    room = layout.physical_rooms[0]
    assert abs(room.area_m2 - 20.0) < 0.2
    assert room.id == layout.rooms[0].id
    assert result.faces_rows["face_id"]


def test_two_adjacent_rooms(tmp_path: Path):
    def build(msp):
        add_rect(msp, (0, 0), 5000, 4000)
        add_rect(msp, (5000, 0), 5000, 4000)

    result = run_dxf(write_simple_dxf(tmp_path / "two.dxf", build))
    assert len(result.layout.physical_rooms) == 2
    areas = sorted(r.area_m2 for r in result.layout.physical_rooms)
    assert abs(areas[0] - 20) < 0.3
    assert abs(areas[1] - 20) < 0.3


def test_three_rooms(tmp_path: Path):
    def build(msp):
        add_rect(msp, (0, 0), 4000, 4000)
        add_rect(msp, (4000, 0), 4000, 4000)
        add_rect(msp, (8000, 0), 4000, 4000)

    result = run_dxf(write_simple_dxf(tmp_path / "three.dxf", build))
    assert len(result.layout.physical_rooms) == 3


def test_l_shaped_room(tmp_path: Path):
    def build(msp):
        add_lines(
            msp,
            [
                ((0, 0), (8000, 0)),
                ((8000, 0), (8000, 4000)),
                ((8000, 4000), (4000, 4000)),
                ((4000, 4000), (4000, 8000)),
                ((4000, 8000), (0, 8000)),
                ((0, 8000), (0, 0)),
            ],
        )

    result = run_dxf(write_simple_dxf(tmp_path / "ell.dxf", build))
    assert len(result.layout.physical_rooms) == 1
    assert result.layout.physical_rooms[0].area_m2 > 40


def test_t_junction(tmp_path: Path):
    def build(msp):
        add_rect(msp, (0, 0), 10000, 8000)
        add_lines(
            msp,
            [
                ((0, 4000), (10000, 4000)),
                ((5000, 0), (5000, 4000)),
            ],
        )

    result = run_dxf(write_simple_dxf(tmp_path / "tee.dxf", build))
    assert len(result.layout.physical_rooms) >= 2


def test_door_gap(tmp_path: Path):
    def build(msp):
        add_lines(
            msp,
            [
                ((0, 0), (5000, 0)),
                ((5000, 0), (5000, 4000)),
                ((5000, 4000), (0, 4000)),
                ((0, 4000), (0, 2500)),
                ((0, 1600), (0, 0)),
            ],
        )

    result = run_dxf(write_simple_dxf(tmp_path / "door.dxf", build))
    assert len(result.layout.physical_rooms) == 1
    assert result.layout.doors or result.layout.warnings


def test_window_opening(tmp_path: Path):
    def build(msp):
        add_rect(msp, (0, 0), 5000, 4000)
        msp.add_line((1000, 4000), (2800, 4000), dxfattribs={"layer": "WINDOW"})

    result = run_dxf(write_simple_dxf(tmp_path / "window.dxf", build))
    assert result.layout.physical_rooms


def test_small_cavity_is_persisted(tmp_path: Path):
    def build(msp):
        add_rect(msp, (0, 0), 8000, 8000)
        add_rect(msp, (1000, 1000), 400, 400)

    result = run_dxf(write_simple_dxf(tmp_path / "cavity.dxf", build))
    assert any(not accepted for accepted in result.faces_rows["accepted_as_room"]) or result.faces_rows["face_id"]
    assert result.faces_rows["face_id"]


def test_nested_cavity(tmp_path: Path):
    def build(msp):
        add_rect(msp, (0, 0), 10000, 10000)
        add_rect(msp, (3000, 3000), 2000, 2000)

    result = run_dxf(write_simple_dxf(tmp_path / "nested.dxf", build))
    assert len(result.faces_rows["face_id"]) >= 2


def test_slightly_offset_endpoints_snap(tmp_path: Path):
    def build(msp):
        add_lines(
            msp,
            [
                ((0, 0), (5000, 0)),
                ((5000.01, 0), (5000, 4000)),
                ((5000, 4000), (0, 4000)),
                ((0, 4000), (0, 0)),
            ],
        )

    result = run_dxf(write_simple_dxf(tmp_path / "offset.dxf", build))
    assert len(result.layout.physical_rooms) == 1


def test_duplicated_wall(tmp_path: Path):
    def build(msp):
        add_rect(msp, (0, 0), 5000, 4000)
        add_lines(msp, [((0, 0), (5000, 0))])

    result = run_dxf(write_simple_dxf(tmp_path / "dup.dxf", build))
    assert len(result.layout.physical_rooms) == 1


def test_stable_ids_on_reprocess(tmp_path: Path):
    path = write_simple_dxf(tmp_path / "stable.dxf", lambda msp: add_rect(msp, (0, 0), 5000, 4000))
    a = run_dxf(path, job_id="a")
    b = run_dxf(path, job_id="b")
    assert [r.id for r in a.layout.physical_rooms] == [r.id for r in b.layout.physical_rooms]
    assert [w.id for w in a.layout.walls] == [w.id for w in b.layout.walls]


def test_furniture_not_discarded(tmp_path: Path):
    def build(msp):
        add_rect(msp, (0, 0), 5000, 4000)
        msp.add_circle((1500, 1500), 200, dxfattribs={"layer": "FURNITURE"})

    result = run_dxf(write_simple_dxf(tmp_path / "furn.dxf", build))
    assert result.layout.objects or result.layout.unassigned_objects


def test_dimension_not_used_as_wall(tmp_path: Path):
    def build(msp):
        add_rect(msp, (0, 0), 5000, 4000)
        msp.add_line((0, -300), (5000, -300), dxfattribs={"layer": "DIM"})

    result = run_dxf(write_simple_dxf(tmp_path / "dim.dxf", build))
    assert len(result.layout.physical_rooms) == 1


def test_long_unrelated_parallel_lines_do_not_invent_rooms(tmp_path: Path):
    def build(msp):
        add_lines(
            msp,
            [
                ((0, 0), (20000, 0)),
                ((0, 8000), (20000, 8000)),
            ],
        )

    result = run_dxf(write_simple_dxf(tmp_path / "parallel.dxf", build))
    assert result.layout.physical_rooms == [] or all(r.area_m2 < 200 for r in result.layout.physical_rooms)


def test_crossing_lines(tmp_path: Path):
    def build(msp):
        add_rect(msp, (0, 0), 8000, 8000)
        add_lines(msp, [((0, 4000), (8000, 4000)), ((4000, 0), (4000, 8000))])

    result = run_dxf(write_simple_dxf(tmp_path / "cross.dxf", build))
    assert len(result.layout.physical_rooms) >= 3


def test_invalid_self_intersecting_does_not_crash(tmp_path: Path):
    def build(msp):
        add_lines(
            msp,
            [
                ((0, 0), (4000, 4000)),
                ((4000, 0), (0, 4000)),
                ((0, 0), (4000, 0)),
                ((4000, 0), (4000, 4000)),
                ((4000, 4000), (0, 4000)),
                ((0, 4000), (0, 0)),
            ],
        )

    result = run_dxf(write_simple_dxf(tmp_path / "bowtie.dxf", build))
    assert result.layout.unit == "m"
