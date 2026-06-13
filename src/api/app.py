"""FastAPI application factory for the Academic Writing Auditor."""

from fastapi import FastAPI

from src.api.error_handlers import register_handlers
from src.api.routes.analyze import router as analyze_router
from src.api.routes.health import router as health_router

_DESCRIPTION = (
    "Academic Writing Auditor — probabilistic writing quality analysis. "
    "Results reflect writing style signals and do not assert AI authorship."
)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    Returns:
        A fully configured FastAPI instance with routes and exception handlers.
    """
    application = FastAPI(
        title="Academic Writing Auditor",
        description=_DESCRIPTION,
        version="0.1.0",
    )
    application.include_router(health_router)
    application.include_router(analyze_router)
    register_handlers(application)
    return application


app = create_app()
