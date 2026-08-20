from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from packages.common.errors import AppError, HTTP_STATUS
from packages.contracts.dto.common import ErrorBodyDto, ErrorEnvelopeDto
from packages.contracts.enums import ErrorCode

def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
        body = ErrorEnvelopeDto(
            success=False,
            error=ErrorBodyDto(code=exc.code, detail=exc.detail, fields=exc.fields),
        )
        return JSONResponse(status_code=exc.http_status, content=body.model_dump())

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = {}
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()))
            fields[loc] = err.get("msg", "invalid")
        body = ErrorEnvelopeDto(
            success=False,
            error=ErrorBodyDto(
                code=ErrorCode.INVALID_REQUEST,
                detail="Request validation failed",
                fields=fields,
            ),
        )
        return JSONResponse(status_code=400, content=body.model_dump())

    @app.exception_handler(ValidationError)
    async def pydantic_handler(_request: Request, exc: ValidationError) -> JSONResponse:
        body = ErrorEnvelopeDto(
            success=False,
            error=ErrorBodyDto(
                code=ErrorCode.INVALID_REQUEST,
                detail=str(exc),
                fields={},
            ),
        )
        return JSONResponse(status_code=400, content=body.model_dump())

    @app.exception_handler(Exception)
    async def unhandled(_request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, StarletteHTTPException):
            raise exc
        body = ErrorEnvelopeDto(
            success=False,
            error=ErrorBodyDto(
                code=ErrorCode.INTERNAL_ERROR,
                detail="internal error",
                fields={},
            ),
        )
        return JSONResponse(status_code=HTTP_STATUS[ErrorCode.INTERNAL_ERROR], content=body.model_dump())
