from __future__ import annotations

from enum import StrEnum


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class UnitSource(StrEnum):
    EXPLICIT = "explicit"
    CAD_METADATA = "cad_metadata"
    FALLBACK = "fallback"


class PrimitiveKind(StrEnum):
    LINE = "line"
    ARC = "arc"
    POLYLINE = "polyline"
    LWPOLYLINE = "lwpolyline"
    CIRCLE = "circle"
    HATCH = "hatch"
    TEXT = "text"
    BLOCK = "block"


class EntityClass(StrEnum):
    WALL = "wall"
    OPENING = "opening"
    FURNITURE = "furniture"
    HATCH = "hatch"
    DIM = "dim"
    TEXT = "text"
    UNKNOWN = "unknown"


class FaceClass(StrEnum):
    INDOOR = "indoor"
    CORRIDOR = "corridor"
    SHAFT = "shaft"
    OUTDOOR = "outdoor"
    CAVITY = "cavity"


class PortalKind(StrEnum):
    DOOR = "door"
    WINDOW = "window"
    OPENING = "opening"
    UNKNOWN = "unknown"


class DecisionSource(StrEnum):
    HEURISTIC = "heuristic"
    MODEL = "model"


class GapDecision(StrEnum):
    CLOSE = "close"
    REJECT = "reject"
    PORTAL = "portal"


class WarningCode(StrEnum):
    UNIT_INFERRED = "UNIT_INFERRED"
    LOW_ENTITY_CONFIDENCE = "LOW_ENTITY_CONFIDENCE"
    LOW_FACE_CONFIDENCE = "LOW_FACE_CONFIDENCE"
    GAP_CLOSE_REJECTED = "GAP_CLOSE_REJECTED"
    INVALID_FACE_REPAIRED = "INVALID_FACE_REPAIRED"
    OBJECT_UNASSIGNED = "OBJECT_UNASSIGNED"
    AMBIGUOUS_ROOM = "AMBIGUOUS_ROOM"
    NO_ROOMS_FOUND = "NO_ROOMS_FOUND"
    CAD_PARSE_WARNING = "CAD_PARSE_WARNING"


class WarningSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    CAD_PARSE_ERROR = "CAD_PARSE_ERROR"
    CONVERSION_ERROR = "CONVERSION_ERROR"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    JOB_NOT_READY = "JOB_NOT_READY"
    JOB_FAILED = "JOB_FAILED"
    LAYOUT_NOT_FOUND = "LAYOUT_NOT_FOUND"
    LAYOUT_REVISION_CONFLICT = "LAYOUT_REVISION_CONFLICT"
    INVALID_FEEDBACK_ACTION = "INVALID_FEEDBACK_ACTION"
    INVALID_TARGET = "INVALID_TARGET"
    EVENT_ALREADY_RETRACTED = "EVENT_ALREADY_RETRACTED"
    MODEL_NOT_AVAILABLE = "MODEL_NOT_AVAILABLE"
    STORAGE_ERROR = "STORAGE_ERROR"
    QUEUE_ERROR = "QUEUE_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class FeedbackAction(StrEnum):
    ACCEPT = "accept"
    DELETE_ROOM = "delete_room"
    RESTORE_FACE = "restore_face"
    MERGE_ROOMS = "merge_rooms"
    SPLIT = "split"
    DRAW_WALL = "draw_wall"
    CONNECT_NODES = "connect_nodes"
    MARK_ENTITY = "mark_entity"
    RENAME_ROOM = "rename_room"
    UNDO = "undo"


class ModelRegistryStatus(StrEnum):
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class LabelSource(StrEnum):
    EXPERT = "expert"
    USER_EXPLICIT = "user_explicit"
    SYNTHETIC = "synthetic"
    WEAK = "weak"


class CadFormat(StrEnum):
    DXF = "dxf"
    DWG = "dwg"
