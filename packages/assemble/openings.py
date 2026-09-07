from __future__ import annotations

from packages.assemble.rooms import PhysicalRoom
from packages.common.spatial_index import SpatialIndex, expand_bounds
from packages.contracts.enums import PortalKind
from packages.graphkit.models import Portal


def attach_openings(
    rooms: tuple[PhysicalRoom, ...],
    portals: tuple[Portal, ...],
) -> tuple[tuple[PhysicalRoom, ...], tuple[Portal, ...], tuple[Portal, ...]]:
    doors = tuple(p for p in portals if p.kind is PortalKind.DOOR)
    windows = tuple(p for p in portals if p.kind is PortalKind.WINDOW)
    door_index = SpatialIndex.build([(p.id, p.geometry) for p in doors])
    window_index = SpatialIndex.build([(p.id, p.geometry) for p in windows])
    updated: list[PhysicalRoom] = []
    for room in rooms:
        buf = room.geometry.buffer(0.08)
        probe = expand_bounds(room.geometry, 0.08)
        door_hits = set(door_index.query(probe))
        window_hits = set(window_index.query(probe))
        door_ids = [p.id for p in doors if p.id in door_hits and buf.intersects(p.geometry)]
        window_ids = [p.id for p in windows if p.id in window_hits and buf.intersects(p.geometry)]
        updated.append(
            PhysicalRoom(
                id=room.id,
                face_id=room.face_id,
                geometry=room.geometry,
                area_m2=room.area_m2,
                perimeter_m=room.perimeter_m,
                center=room.center,
                bbox=room.bbox,
                confidence=room.confidence,
                kind=room.kind,
                neighbor_ids=room.neighbor_ids,
                door_ids=tuple(door_ids),
                window_ids=tuple(window_ids),
                object_ids=room.object_ids,
                name=room.name,
            )
        )
    return tuple(updated), doors, windows
