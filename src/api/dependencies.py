"""FastAPI dependency providers for the Academic Writing Auditor."""

from functools import lru_cache

from src.services.analysis import AnalysisService


@lru_cache(maxsize=1)
def get_analysis_service() -> AnalysisService:
    """Return a process-lifetime singleton AnalysisService.

    ``lru_cache`` ensures the TokenizerService NLTK initialisation runs
    exactly once per process.  Tests override this via
    ``app.dependency_overrides``.

    Returns:
        Shared AnalysisService instance.
    """
    return AnalysisService()
