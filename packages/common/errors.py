"""Domain errors mapped to documented API error codes."""

from __future__ import annotations

from packages.contracts.enums import ErrorCode

HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.INVALID_REQUEST: 400,
    ErrorCode.UNSUPPORTED_FORMAT: 400,
    ErrorCode.FILE_TOO_LARGE: 413,
    ErrorCode.CAD_PARSE_ERROR: 400,
    ErrorCode.CONVERSION_ERROR: 400,
    ErrorCode.JOB_NOT_FOUND: 404,
    ErrorCode.JOB_NOT_READY: 409,
    ErrorCode.JOB_FAILED: 409,
    ErrorCode.LAYOUT_NOT_FOUND: 404,
    ErrorCode.LAYOUT_REVISION_CONFLICT: 409,
    ErrorCode.INVALID_FEEDBACK_ACTION: 400,
    ErrorCode.INVALID_TARGET: 400,
    ErrorCode.EVENT_ALREADY_RETRACTED: 409,
    ErrorCode.MODEL_NOT_AVAILABLE: 404,
    ErrorCode.STORAGE_ERROR: 500,
    ErrorCode.QUEUE_ERROR: 503,
    ErrorCode.INTERNAL_ERROR: 500,
}


class AppError(Exception):
    def __init__(
        self,
        code: ErrorCode,
        detail: str,
        fields: dict[str, str] | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.fields = fields or {}
        self.http_status = http_status if http_status is not None else HTTP_STATUS[code]


class InvalidRequestError(AppError):
    def __init__(self, detail: str, fields: dict[str, str] | None = None) -> None:
        super().__init__(ErrorCode.INVALID_REQUEST, detail, fields)


class UnsupportedFormatError(AppError):
    def __init__(self, detail: str) -> None:
        super().__init__(ErrorCode.UNSUPPORTED_FORMAT, detail)


class FileTooLargeError(AppError):
    def __init__(self, detail: str) -> None:
        super().__init__(ErrorCode.FILE_TOO_LARGE, detail)


class CadParseError(AppError):
    def __init__(self, detail: str) -> None:
        super().__init__(ErrorCode.CAD_PARSE_ERROR, detail)


class ConversionError(AppError):
    def __init__(self, detail: str) -> None:
        super().__init__(ErrorCode.CONVERSION_ERROR, detail)


class JobNotFoundError(AppError):
    def __init__(self, job_id: str) -> None:
        super().__init__(ErrorCode.JOB_NOT_FOUND, f"Job {job_id} was not found")


class JobNotReadyError(AppError):
    def __init__(self, job_id: str, status: str) -> None:
        super().__init__(
            ErrorCode.JOB_NOT_READY,
            f"Job {job_id} is not ready (status={status})",
        )


class JobFailedError(AppError):
    def __init__(self, job_id: str, detail: str) -> None:
        super().__init__(ErrorCode.JOB_FAILED, f"Job {job_id} failed: {detail}")


class LayoutNotFoundError(AppError):
    def __init__(self, job_id: str) -> None:
        super().__init__(ErrorCode.LAYOUT_NOT_FOUND, f"Layout for job {job_id} was not found")


class LayoutRevisionConflictError(AppError):
    def __init__(self, job_id: str, expected: int, actual: int) -> None:
        super().__init__(
            ErrorCode.LAYOUT_REVISION_CONFLICT,
            f"Layout revision conflict for job {job_id}: expected {expected}, actual {actual}",
        )


class InvalidFeedbackActionError(AppError):
    def __init__(self, detail: str) -> None:
        super().__init__(ErrorCode.INVALID_FEEDBACK_ACTION, detail)


class InvalidTargetError(AppError):
    def __init__(self, detail: str, fields: dict[str, str] | None = None) -> None:
        super().__init__(ErrorCode.INVALID_TARGET, detail, fields)


class EventAlreadyRetractedError(AppError):
    def __init__(self, event_id: str) -> None:
        super().__init__(
            ErrorCode.EVENT_ALREADY_RETRACTED,
            f"Event {event_id} has already been retracted",
        )


class StorageError(AppError):
    def __init__(self, detail: str) -> None:
        super().__init__(ErrorCode.STORAGE_ERROR, detail)


class QueueError(AppError):
    def __init__(self, detail: str) -> None:
        super().__init__(ErrorCode.QUEUE_ERROR, detail)
