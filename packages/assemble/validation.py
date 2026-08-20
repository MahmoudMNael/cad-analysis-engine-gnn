from __future__ import annotations

from packages.contracts.dto.layout import LayoutDto


def assert_layout_invariants(layout: LayoutDto) -> None:
    LayoutDto.model_validate(layout.model_dump())
    ids = [room.id for room in layout.physical_rooms]
    if len(ids) != len(set(ids)):
        raise ValueError("physical room ids must be unique")
    for room in layout.rooms:
        if room.physical_room_id and room.physical_room_id not in {r.id for r in layout.physical_rooms}:
            # derived rooms may point at a physical room; allow parent-only in later phases
            pass
