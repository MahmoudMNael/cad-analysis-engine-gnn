"""Admin training is P6. Route exists as a typed contract."""

from __future__ import annotations

from fastapi import APIRouter

from packages.common.errors import AppError
from packages.contracts.dto.models import TrainRequest
from packages.contracts.enums import ErrorCode

router = APIRouter()


@router.post("/admin/train")
def train(body: TrainRequest) -> None:
    raise AppError(ErrorCode.MODEL_NOT_AVAILABLE, "training is not available in P1")
