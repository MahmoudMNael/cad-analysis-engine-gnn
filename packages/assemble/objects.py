from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from shapely.geometry import Point, box as shapely_box
from shapely.geometry.base import BaseGeometry

from packages.assemble.rooms import PhysicalRoom
from packages.common.configuration import Settings
from packages.common.hashing import stable_hash
from packages.common.spatial_index import SpatialIndex, expand_bounds
from packages.contracts.enums import EntityClass
from packages.ingest.models import Primitive
from packages.models_nn.common import EntityScore

_BLOCK_CLUSTER_GAP_M = 0.20


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
    furniture: list[Primitive] = []
    for prim in primitives:
        score = scores.get(prim.id)
        is_furniture = False
        if score and score.predicted_class == EntityClass.FURNITURE.value:
            is_furniture = True
        elif prim.block_name and (score is None or score.predicted_class not in {
            EntityClass.WALL.value,
            EntityClass.OPENING.value,
            EntityClass.TEXT.value,
            EntityClass.DIM.value,
            EntityClass.HATCH.value,
        }):
            is_furniture = True
        if is_furniture:
            furniture.append(prim)

    assigned: list[PlacedObject] = []
    unassigned: list[PlacedObject] = []
    room_index = SpatialIndex.build([(r.id, r.geometry) for r in rooms])
    tol = settings.object_assignment_tolerance_m
    for group in _cluster_furniture(furniture):
        obj = _placed_from_group(group, scores)
        cx, cy = obj.center
        probe = Point(cx, cy)
        room_id = None
        hits = set(room_index.query(expand_bounds(probe, tol)))
        for room in rooms:
            if room.id not in hits:
                continue
            if room.geometry.buffer(tol).contains(probe):
                room_id = room.id
                break
        obj = PlacedObject(
            id=obj.id,
            kind=obj.kind,
            geometry=obj.geometry,
            center=obj.center,
            bbox=obj.bbox,
            layer=obj.layer,
            block_name=obj.block_name,
            room_id=room_id,
            confidence=obj.confidence,
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


def _cluster_furniture(primitives: list[Primitive]) -> list[list[Primitive]]:
    if not primitives:
        return []
    groups: list[list[Primitive]] = []
    by_block: dict[str, list[Primitive]] = defaultdict(list)
    for prim in primitives:
        if prim.block_name:
            by_block[prim.block_name].append(prim)
        else:
            groups.append([prim])
    for members in by_block.values():
        groups.extend(_cluster_nearby(members, _BLOCK_CLUSTER_GAP_M))
    return groups


def _cluster_nearby(members: list[Primitive], gap_m: float) -> list[list[Primitive]]:
    n = len(members)
    if n <= 1:
        return [members] if members else []
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    cell = max(gap_m, 1e-6)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    boxes = [p.bbox.as_tuple() for p in members]
    centers = [((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5) for b in boxes]
    for i, (x, y) in enumerate(centers):
        buckets[(int(x // cell), int(y // cell))].append(i)
    for i, (x, y) in enumerate(centers):
        cx, cy = int(x // cell), int(y // cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((cx + dx, cy + dy), ()):
                    if j <= i:
                        continue
                    if _boxes_close(boxes[i], boxes[j], gap_m):
                        union(i, j)
    clustered: dict[int, list[Primitive]] = defaultdict(list)
    for i, prim in enumerate(members):
        clustered[find(i)].append(prim)
    return list(clustered.values())


def _boxes_close(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    gap_m: float,
) -> bool:
    gap_x = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    gap_y = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return gap_x <= gap_m and gap_y <= gap_m


def _placed_from_group(group: list[Primitive], scores: dict[str, EntityScore]) -> PlacedObject:
    if len(group) == 1:
        prim = group[0]
        c = prim.geometry.centroid
        conf = scores[prim.id].confidence if prim.id in scores else 0.5
        return PlacedObject(
            id=prim.id,
            kind="furniture",
            geometry=prim.geometry,
            center=(float(c.x), float(c.y)),
            bbox=prim.bbox.as_tuple(),
            layer=prim.layer,
            block_name=prim.block_name,
            room_id=None,
            confidence=conf,
        )
    minx = min(p.bbox.minx for p in group)
    miny = min(p.bbox.miny for p in group)
    maxx = max(p.bbox.maxx for p in group)
    maxy = max(p.bbox.maxy for p in group)
    center = ((minx + maxx) * 0.5, (miny + maxy) * 0.5)
    object_id = stable_hash("object", [center], {"members": ",".join(sorted(p.id for p in group))})
    confs = [scores[p.id].confidence for p in group if p.id in scores]
    primary = max(group, key=lambda p: p.length)
    return PlacedObject(
        id=object_id,
        kind="furniture",
        geometry=shapely_box(minx, miny, maxx, maxy),
        center=center,
        bbox=(minx, miny, maxx, maxy),
        layer=primary.layer,
        block_name=primary.block_name,
        room_id=None,
        confidence=max(confs) if confs else 0.5,
    )
