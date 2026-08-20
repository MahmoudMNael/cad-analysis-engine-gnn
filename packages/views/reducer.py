"""P1 derived view is a pass-through of physical rooms. Split/merge live in P2."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from packages.contracts.dto.layout import LayoutDto


def reduce_view(physical_layout: LayoutDto, events: Sequence[Any]) -> LayoutDto:
    if events:
        raise RuntimeError("view events are implemented in P2")
    dumped = physical_layout.model_dump()
    dumped["rooms"] = dumped["physical_rooms"]
    return LayoutDto.model_validate(dumped)
