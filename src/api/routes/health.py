"""Health check endpoint for the Academic Writing Auditor API."""

from fastapi import APIRouter
from pydantic import BaseModel

_VERSION = "0.1.0"

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response payload.

    Attributes:
        status: Service liveness state. Always "ok" when reachable.
        version: Application version string.
    """

    status: str
    version: str


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service liveness information.

    Returns:
        HealthResponse with status "ok" and the current application version.
    """
    return HealthResponse(status="ok", version=_VERSION)
