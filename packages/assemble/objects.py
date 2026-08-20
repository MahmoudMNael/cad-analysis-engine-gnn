from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry.base import BaseGeometry

from packages.assemble.rooms import PhysicalRoom
from packages.common.configuration import Settings
from packages.contracts.enums import EntityClass
from packages.ingest.models import Primitive
from packages.models_nn.common import EntityScore


@dataclass(frozen=True)
class PlacedObject:
    id: str
    kind: str
    geometry: BaseGeometry
    center: tuple[float, float]
    bbox: tuple[float, float, float, float]
    layer: str | None
    block_name: str | None
    room_id: str | None
    confidence: float


def assemble_objects(
    primitives: tuple[Primitive, ...],
    scores: dict[str, EntityScore],
    rooms: tuple[PhysicalRoom, ...],
    settings: Settings,
) -> tuple[tuple[PlacedObject, ...], tuple[PlacedObject, ...], tuple[PhysicalRoom, ...]]:
    assigned: list[PlacedObject] = []
    unassigned: list[PlacedObject] = []
    for prim in primitives:
        score = scores.get(prim.id)
        furniture = False
        if score and score.predicted_class == EntityClass.FURNITURE.value:
            furniture = True
        elif prim.block_name and (score is None or score.predicted_class not in {
            EntityClass.WALL.value,
            EntityClass.OPENING.value,
            EntityClass.TEXT.value,
            EntityClass.DIM.value,
            EntityClass.HATCH.value,
        }):
            furniture = True
        if not furniture:
            continue
        c = prim.geometry.centroid
        room_id = None
        for room in rooms:
            if room.geometry.buffer(settings.object_assignment_tolerance_m).contains(c):
                room_id = room.id
                break
        obj = PlacedObject(
            id=prim.id,
            kind="furniture",
            geometry=prim.geometry,
            center=(float(c.x), float(c.y)),
            bbox=prim.bbox.as_tuple(),
            layer=prim.layer,
            block_name=prim.block_name,
            room_id=room_id,
            confidence=score.confidence if score else 0.5,
        )
        if room_id:
            assigned.append(obj)
        else:
            unassigned.append(obj)
    rooms_updated = []
    by_room: dict[str, list[str]] = {}
    for obj in assigned:
        if obj.room_id:
            by_room.setdefault(obj.room_id, []).append(obj.id)
    for room in rooms:
        rooms_updated.append(
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
                door_ids=room.door_ids,
                window_ids=room.window_ids,
                object_ids=tuple(by_room.get(room.id, [])),
                name=room.name,
            )
        )
    return tuple(assigned), tuple(unassigned), tuple(rooms_updated)
