from __future__ import annotations

from packages.assemble.objects import PlacedObject
from packages.assemble.polygonize import FaceSet
from packages.assemble.rooms import PhysicalRoom
from packages.contracts.dto.common import WarningDto
from packages.contracts.enums import UnitSource, WarningCode, WarningSeverity
from packages.ingest.models import Document
from packages.models_nn.common import EntityScore, FaceScore, GapScore


def compile_warnings(
    *,
    document: Document,
    ingest_warnings: list[WarningDto],
    entity_scores: dict[str, EntityScore],
    gap_scores: dict[str, GapScore],
    face_scores: dict[str, FaceScore],
    faces: FaceSet,
    rooms: tuple[PhysicalRoom, ...],
    unassigned: tuple[PlacedObject, ...],
    entity_threshold: float,
    face_threshold: float,
) -> list[WarningDto]:
    warnings = list(ingest_warnings)
    if document.unit_source is UnitSource.FALLBACK:
        warnings.append(
            WarningDto(
                code=WarningCode.UNIT_INFERRED,
                severity=WarningSeverity.INFO,
                message="CAD unit metadata was untrustworthy; fallback unit was used",
                target_ids=[],
            )
        )
    low_entities = [s.target_id for s in entity_scores.values() if s.confidence < entity_threshold]
    if low_entities:
        warnings.append(
            WarningDto(
                code=WarningCode.LOW_ENTITY_CONFIDENCE,
                severity=WarningSeverity.WARNING,
                message="One or more primitives have low entity confidence",
                target_ids=low_entities[:50],
            )
        )
    low_faces = [s.face_id for s in face_scores.values() if s.confidence < face_threshold]
    if low_faces:
        warnings.append(
            WarningDto(
                code=WarningCode.LOW_FACE_CONFIDENCE,
                severity=WarningSeverity.WARNING,
                message="One or more faces have low classification confidence",
                target_ids=low_faces[:50],
            )
        )
    rejected = [s.pair_id for s in gap_scores.values() if s.decision.value == "reject"]
    if rejected:
        warnings.append(
            WarningDto(
                code=WarningCode.GAP_CLOSE_REJECTED,
                severity=WarningSeverity.INFO,
                message="One or more gap candidates were rejected",
                target_ids=rejected[:50],
            )
        )
    repaired = [f.id for f in faces.faces if f.repaired]
    if repaired:
        warnings.append(
            WarningDto(
                code=WarningCode.INVALID_FACE_REPAIRED,
                severity=WarningSeverity.WARNING,
                message="Invalid faces were repaired deterministically",
                target_ids=repaired,
            )
        )
    if unassigned:
        warnings.append(
            WarningDto(
                code=WarningCode.OBJECT_UNASSIGNED,
                severity=WarningSeverity.INFO,
                message="Objects were preserved without a containing room",
                target_ids=[o.id for o in unassigned],
            )
        )
    if not rooms:
        warnings.append(
            WarningDto(
                code=WarningCode.NO_ROOMS_FOUND,
                severity=WarningSeverity.WARNING,
                message="No physical rooms were assembled",
                target_ids=[],
            )
        )
    ambiguous = [r.id for r in rooms if r.confidence < face_threshold]
    if ambiguous:
        warnings.append(
            WarningDto(
                code=WarningCode.AMBIGUOUS_ROOM,
                severity=WarningSeverity.WARNING,
                message="One or more rooms have ambiguous face confidence",
                target_ids=ambiguous,
            )
        )
    return warnings
