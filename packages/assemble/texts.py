from __future__ import annotations

from dataclasses import dataclass, replace

from packages.assemble.rooms import PhysicalRoom
from packages.common.spatial_index import SpatialIndex
from packages.contracts.enums import EntityClass
from packages.ingest.models import Primitive
from packages.models_nn.common import EntityScore


@dataclass(frozen=True)
class LayoutText:
    id: str
    content: str
    position: tuple[float, float]
    bbox: tuple[float, float, float, float]
    rotation: float
    layer: str | None
    confidence: float
    room_id: str | None


def assemble_texts(
    primitives: tuple[Primitive, ...],
    scores: dict[str, EntityScore],
    rooms: tuple[PhysicalRoom, ...],
) -> tuple[tuple[LayoutText, ...], tuple[PhysicalRoom, ...]]:
    texts: list[LayoutText] = []
    names: dict[str, str] = {}
    room_index = SpatialIndex.build([(r.id, r.geometry) for r in rooms])
    for prim in primitives:
        score = scores.get(prim.id)
        is_text = prim.kind.value == "text" or (score and score.predicted_class == EntityClass.TEXT.value)
        if not is_text:
            continue
        content = str(prim.attributes.get("content") or "")
        c = prim.geometry.centroid
        room_id = None
        hits = set(room_index.query(prim.geometry)) | set(room_index.query(c))
        for room in rooms:
            if room.id not in hits:
                continue
            if room.geometry.contains(c) or room.geometry.intersects(prim.geometry):
                room_id = room.id
                if content.strip() and not _looks_like_dimension(content) and room.id not in names:
                    names[room.id] = content.strip()
                break
        texts.append(
            LayoutText(
                id=prim.id,
                content=content,
                position=(float(c.x), float(c.y)),
                bbox=prim.bbox.as_tuple(),
                rotation=float(prim.attributes.get("rotation") or 0.0),
                layer=prim.layer,
                confidence=score.confidence if score else 1.0,
                room_id=room_id,
            )
        )
    updated = []
    for room in rooms:
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
                door_ids=room.door_ids,
                window_ids=room.window_ids,
                object_ids=room.object_ids,
                name=names.get(room.id, room.name),
            )
        )
    return tuple(texts), tuple(updated)


def _looks_like_dimension(text: str) -> bool:
    stripped = text.strip().replace(",", ".")
    if not stripped:
        return True
    try:
        float(stripped.replace("m", "").replace("mm", "").replace("'", ""))
        return True
    except ValueError:
        return False
