from __future__ import annotations

from packages.assemble.polygonize import Face
from packages.contracts.enums import FaceClass
from packages.models_nn.common import FaceScore


ROOM_FACE_CLASSES = {FaceClass.INDOOR.value, FaceClass.CORRIDOR.value}


def is_room_face(score: FaceScore, face: Face) -> bool:
    if not face.valid or face.discarded_reason is not None:
        return False
    return score.predicted_class in ROOM_FACE_CLASSES and face.area > 0
