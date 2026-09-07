"""Named DXF builders shared by synthetic tests and pipeline goldens."""

from __future__ import annotations

from collections.abc import Callable

from tests.helpers.dxf_builder import add_lines, add_rect

Builder = Callable


def build_single_rectangle(msp) -> None:
    add_rect(msp, (0, 0), 5000, 4000)


def build_two_adjacent_rooms(msp) -> None:
    add_rect(msp, (0, 0), 5000, 4000)
    add_rect(msp, (5000, 0), 5000, 4000)


def build_three_rooms(msp) -> None:
    add_rect(msp, (0, 0), 4000, 4000)
    add_rect(msp, (4000, 0), 4000, 4000)
    add_rect(msp, (8000, 0), 4000, 4000)


def build_l_shaped_room(msp) -> None:
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


def build_t_junction(msp) -> None:
    add_rect(msp, (0, 0), 10000, 8000)
    add_lines(
        msp,
        [
            ((0, 4000), (10000, 4000)),
            ((5000, 0), (5000, 4000)),
        ],
    )


def build_door_gap(msp) -> None:
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


def build_window_opening(msp) -> None:
    add_rect(msp, (0, 0), 5000, 4000)
    msp.add_line((1000, 4000), (2800, 4000), dxfattribs={"layer": "WINDOW"})


def build_small_cavity(msp) -> None:
    add_rect(msp, (0, 0), 8000, 8000)
    add_rect(msp, (1000, 1000), 400, 400)


def build_nested_cavity(msp) -> None:
    add_rect(msp, (0, 0), 10000, 10000)
    add_rect(msp, (3000, 3000), 2000, 2000)


def build_slightly_offset_endpoints(msp) -> None:
    add_lines(
        msp,
        [
            ((0, 0), (5000, 0)),
            ((5000.01, 0), (5000, 4000)),
            ((5000, 4000), (0, 4000)),
            ((0, 4000), (0, 0)),
        ],
    )


def build_duplicated_wall(msp) -> None:
    add_rect(msp, (0, 0), 5000, 4000)
    add_lines(msp, [((0, 0), (5000, 0))])


def build_furniture(msp) -> None:
    add_rect(msp, (0, 0), 5000, 4000)
    msp.add_circle((1500, 1500), 200, dxfattribs={"layer": "FURNITURE"})


def build_dimension(msp) -> None:
    add_rect(msp, (0, 0), 5000, 4000)
    msp.add_line((0, -300), (5000, -300), dxfattribs={"layer": "DIM"})


def build_long_parallel(msp) -> None:
    add_lines(
        msp,
        [
            ((0, 0), (20000, 0)),
            ((0, 8000), (20000, 8000)),
        ],
    )


def build_crossing_lines(msp) -> None:
    add_rect(msp, (0, 0), 8000, 8000)
    add_lines(msp, [((0, 4000), (8000, 4000)), ((4000, 0), (4000, 8000))])


def build_invalid_self_intersecting(msp) -> None:
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


PIPELINE_CASES: tuple[tuple[str, Builder], ...] = (
    ("single_rectangle", build_single_rectangle),
    ("two_adjacent_rooms", build_two_adjacent_rooms),
    ("three_rooms", build_three_rooms),
    ("l_shaped_room", build_l_shaped_room),
    ("t_junction", build_t_junction),
    ("door_gap", build_door_gap),
    ("window_opening", build_window_opening),
    ("small_cavity", build_small_cavity),
    ("nested_cavity", build_nested_cavity),
    ("slightly_offset_endpoints", build_slightly_offset_endpoints),
    ("duplicated_wall", build_duplicated_wall),
    ("furniture", build_furniture),
    ("dimension", build_dimension),
    ("long_parallel", build_long_parallel),
    ("crossing_lines", build_crossing_lines),
    ("invalid_self_intersecting", build_invalid_self_intersecting),
)
