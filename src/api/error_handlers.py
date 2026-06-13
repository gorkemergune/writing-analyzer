"""Exception handlers for the Academic Writing Auditor API."""

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.services.language_detector import LanguageDetectionError

_log = logging.getLogger(__name__)


def register_handlers(app: FastAPI) -> None:
    """Attach exception handlers to the FastAPI application.

    Args:
        app: The FastAPI application instance to configure.
    """

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors(), "body": exc.body},
        )

    @app.exception_handler(LanguageDetectionError)
    async def _language_detection_handler(
        request: Request, exc: LanguageDetectionError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc)},
        )

    @app.exception_handler(Exception)
    async def _generic_handler(request: Request, exc: Exception) -> JSONResponse:
        _log.exception("Unhandled error processing %s", request.url)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error."},
        )
