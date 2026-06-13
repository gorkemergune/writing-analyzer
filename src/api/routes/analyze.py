"""Analysis endpoint for the Academic Writing Auditor API."""

from typing import Annotated

from fastapi import APIRouter, Depends

from src.api.dependencies import get_analysis_service
from src.models.request import AnalysisRequest
from src.models.response import AnalysisReport
from src.services.analysis import AnalysisService

router = APIRouter()


@router.post("/api/v1/analyze", response_model=AnalysisReport)
def analyze(
    request: AnalysisRequest,
    service: Annotated[AnalysisService, Depends(get_analysis_service)],
) -> AnalysisReport:
    """Run the full writing analysis pipeline on the submitted text.

    Args:
        request: Validated request body containing text and options.
        service: Injected AnalysisService singleton.

    Returns:
        AnalysisReport containing scores, highlights, and suggestions.
    """
    return service.analyze(request)
