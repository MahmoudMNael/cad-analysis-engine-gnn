from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Polygon

from packages.assemble.face_filters import is_room_face
from packages.assemble.polygonize import Face, FaceSet
from packages.common.geometry import aspect_ratio
from packages.contracts.enums import FaceClass
from packages.models_nn.common import FaceScore


@dataclass(frozen=True)
class PhysicalRoom:
    id: str
    face_id: str
    geometry: Polygon
    area_m2: float
    perimeter_m: float
    center: tuple[float, float]
    bbox: tuple[float, float, float, float]
    confidence: float
    kind: str
    neighbor_ids: tuple[str, ...]
    door_ids: tuple[str, ...]
    window_ids: tuple[str, ...]
    object_ids: tuple[str, ...]
    name: str | None = None


def assemble_physical_rooms(
    faces: FaceSet,
    scores: dict[str, FaceScore],
) -> tuple[PhysicalRoom, ...]:
    accepted: list[tuple[Face, FaceScore]] = []
    for face in faces.faces:
        score = scores.get(face.id)
        if score is None:
            continue
        if is_room_face(score, face):
            accepted.append((face, score))
    neighbors: dict[str, list[str]] = {face.id: [] for face, _ in accepted}
    for i, (a, _) in enumerate(accepted):
        for b, _ in accepted[i + 1 :]:
            if a.geometry.boundary.intersects(b.geometry.boundary.buffer(1e-4)):
                neighbors[a.id].append(b.id)
                neighbors[b.id].append(a.id)
    rooms: list[PhysicalRoom] = []
    for face, score in accepted:
        minx, miny, maxx, maxy = face.geometry.bounds
        c = face.geometry.centroid
        kind = score.predicted_class
        if kind not in {FaceClass.INDOOR.value, FaceClass.CORRIDOR.value}:
            continue
        rooms.append(
            PhysicalRoom(
                id=face.id,
                face_id=face.id,
                geometry=face.geometry,
                area_m2=float(face.area),
                perimeter_m=float(face.perimeter),
                center=(float(c.x), float(c.y)),
                bbox=(float(minx), float(miny), float(maxx), float(maxy)),
                confidence=score.confidence,
                kind=kind,
                neighbor_ids=tuple(sorted(neighbors.get(face.id, []))),
                door_ids=(),
                window_ids=(),
                object_ids=(),
            )
        )
    return tuple(rooms)
