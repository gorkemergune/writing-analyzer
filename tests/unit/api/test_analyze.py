"""Tests for the POST /api/v1/analyze endpoint."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.dependencies import get_analysis_service
from src.models.enums import DocumentType, Language, RiskLevel
from src.models.request import AnalysisRequest
from src.models.response import (
    AcademicRiskScore,
    AnalysisReport,
    BurstinessResult,
    ClicheResult,
    ComponentScores,
    ReadabilityResult,
    RepetitionResult,
    SentenceStats,
    TransitionResult,
    WordStats,
)
from src.services.language_detector import LanguageDetectionError

_VALID_TEXT = (
    "The study of language reveals fundamental aspects of human cognition and thought."
)

_VALID_BODY: dict = {
    "text": _VALID_TEXT,
    "document_type": "essay",
}

_STUB_REPORT = AnalysisReport(
    language=Language.ENGLISH,
    document_type=DocumentType.ESSAY,
    word_stats=WordStats(
        total_words=14,
        unique_words=13,
        lexical_diversity=0.93,
        avg_word_length=5.2,
    ),
    sentence_stats=SentenceStats(
        total_sentences=1,
        avg_sentence_length=14.0,
        sentence_length_variance=0.0,
        min_sentence_length=14,
        max_sentence_length=14,
    ),
    repetition=RepetitionResult(
        repeated_words=[],
        repeated_phrases=[],
        repeated_openings=[],
        repetition_score=0.0,
    ),
    transitions=TransitionResult(
        transition_count=0,
        unique_transitions=[],
        repeated_transitions=[],
        transition_density=0.0,
        transition_score=0.0,
    ),
    burstiness=BurstinessResult(
        burstiness_score=0.5,
        burstiness_value=0.0,
        classification="neutral",
    ),
    readability=ReadabilityResult(
        readability_score=62.0,
        grade_level="Grade 9",
        classification="standard",
    ),
    cliches=ClicheResult(
        detected_cliches=[],
        cliche_count=0,
        cliche_density=0.0,
        cliche_score=0.0,
    ),
    academic_risk=AcademicRiskScore(
        overall_score=12.0,
        risk_level=RiskLevel.LOW,
        confidence=0.75,
        component_scores=ComponentScores(
            repetition=5.0,
            transition_overuse=0.0,
            low_burstiness=25.0,
            lexical_poverty=3.0,
            cliche_density=0.0,
            readability=18.0,
        ),
        explanations=[],
    ),
    highlights=[],
    suggestions=[],
    processing_time_ms=10.0,
)


class _StubService:
    """Test double that always returns _STUB_REPORT."""

    def analyze(self, request: AnalysisRequest) -> AnalysisReport:
        """Return the fixed stub report regardless of input."""
        return _STUB_REPORT


class _RaisingService:
    """Test double that raises a configured exception on analyze."""

    def __init__(self, exc: Exception) -> None:
        """Store the exception to raise."""
        self._exc = exc

    def analyze(self, request: AnalysisRequest) -> AnalysisReport:
        """Raise the configured exception."""
        raise self._exc


_client = TestClient(app, raise_server_exceptions=False)


class TestAnalyzeSuccess:
    """POST /api/v1/analyze returns 200 and a valid report for well-formed input."""

    def setup_method(self):
        """Wire stub service into the dependency graph."""
        app.dependency_overrides[get_analysis_service] = lambda: _StubService()

    def teardown_method(self):
        """Restore real dependencies after each test."""
        app.dependency_overrides.clear()

    def test_returns_200_on_valid_request(self):
        """Valid request body yields HTTP 200."""
        response = _client.post("/api/v1/analyze", json=_VALID_BODY)
        assert response.status_code == 200

    def test_response_contains_language(self):
        """Response body includes a language field."""
        response = _client.post("/api/v1/analyze", json=_VALID_BODY)
        assert "language" in response.json()

    def test_response_language_value(self):
        """Language in response matches the stub report."""
        response = _client.post("/api/v1/analyze", json=_VALID_BODY)
        assert response.json()["language"] == "en"

    def test_response_contains_academic_risk(self):
        """Response body includes an academic_risk object."""
        response = _client.post("/api/v1/analyze", json=_VALID_BODY)
        assert "academic_risk" in response.json()

    def test_response_risk_level_is_string(self):
        """risk_level is serialized as a string enum value."""
        response = _client.post("/api/v1/analyze", json=_VALID_BODY)
        assert isinstance(response.json()["academic_risk"]["risk_level"], str)

    def test_response_processing_time_non_negative(self):
        """processing_time_ms is a non-negative number."""
        response = _client.post("/api/v1/analyze", json=_VALID_BODY)
        assert response.json()["processing_time_ms"] >= 0.0

    def test_response_highlights_is_list(self):
        """highlights field is a list."""
        response = _client.post("/api/v1/analyze", json=_VALID_BODY)
        assert isinstance(response.json()["highlights"], list)

    def test_response_suggestions_is_list(self):
        """suggestions field is a list."""
        response = _client.post("/api/v1/analyze", json=_VALID_BODY)
        assert isinstance(response.json()["suggestions"], list)

    def test_content_type_json(self):
        """Response Content-Type is application/json."""
        response = _client.post("/api/v1/analyze", json=_VALID_BODY)
        assert "application/json" in response.headers["content-type"]

    def test_optional_language_override_accepted(self):
        """Explicitly providing language is accepted without error."""
        body = {**_VALID_BODY, "language": "en"}
        response = _client.post("/api/v1/analyze", json=body)
        assert response.status_code == 200

    def test_optional_language_none_accepted(self):
        """Omitting language field triggers auto-detection path without error."""
        body = {"text": _VALID_TEXT}
        response = _client.post("/api/v1/analyze", json=body)
        assert response.status_code == 200


class TestAnalyzeValidation:
    """POST /api/v1/analyze returns 422 for invalid request bodies."""

    def setup_method(self):
        """Wire stub service so validation failures are the only source of 422."""
        app.dependency_overrides[get_analysis_service] = lambda: _StubService()

    def teardown_method(self):
        """Restore real dependencies after each test."""
        app.dependency_overrides.clear()

    def test_422_on_missing_body(self):
        """Omitting the request body returns 422."""
        response = _client.post("/api/v1/analyze")
        assert response.status_code == 422

    def test_422_on_text_too_short(self):
        """Text shorter than 50 characters returns 422."""
        response = _client.post(
            "/api/v1/analyze", json={"text": "Too short."}
        )
        assert response.status_code == 422

    def test_422_on_empty_text(self):
        """Empty string for text returns 422."""
        response = _client.post("/api/v1/analyze", json={"text": ""})
        assert response.status_code == 422

    def test_422_on_invalid_document_type(self):
        """Unknown document_type value returns 422."""
        body = {**_VALID_BODY, "document_type": "not_a_real_type"}
        response = _client.post("/api/v1/analyze", json=body)
        assert response.status_code == 422

    def test_422_response_has_detail_key(self):
        """Validation error response body contains a detail key."""
        response = _client.post(
            "/api/v1/analyze", json={"text": "short"}
        )
        assert "detail" in response.json()


class TestAnalyzeErrorHandling:
    """POST /api/v1/analyze maps service exceptions to appropriate HTTP codes."""

    def teardown_method(self):
        """Restore real dependencies after each test."""
        app.dependency_overrides.clear()

    def test_422_on_language_detection_error(self):
        """LanguageDetectionError raised by the service maps to 422."""
        app.dependency_overrides[get_analysis_service] = lambda: _RaisingService(
            LanguageDetectionError("could not detect language")
        )
        response = _client.post("/api/v1/analyze", json=_VALID_BODY)
        assert response.status_code == 422

    def test_422_language_error_has_detail(self):
        """LanguageDetectionError 422 response includes a detail message."""
        app.dependency_overrides[get_analysis_service] = lambda: _RaisingService(
            LanguageDetectionError("could not detect language")
        )
        response = _client.post("/api/v1/analyze", json=_VALID_BODY)
        assert "detail" in response.json()

    def test_500_on_unexpected_runtime_error(self):
        """Unhandled RuntimeError from the service returns 500."""
        app.dependency_overrides[get_analysis_service] = lambda: _RaisingService(
            RuntimeError("something went wrong")
        )
        response = _client.post("/api/v1/analyze", json=_VALID_BODY)
        assert response.status_code == 500

    def test_500_detail_is_generic(self):
        """The 500 response body does not leak the internal exception message."""
        app.dependency_overrides[get_analysis_service] = lambda: _RaisingService(
            RuntimeError("database connection lost")
        )
        response = _client.post("/api/v1/analyze", json=_VALID_BODY)
        body = response.json()
        assert "database connection lost" not in str(body)

    def test_500_response_has_detail_key(self):
        """The 500 response body contains a detail key."""
        app.dependency_overrides[get_analysis_service] = lambda: _RaisingService(
            ValueError("unexpected")
        )
        response = _client.post("/api/v1/analyze", json=_VALID_BODY)
        assert "detail" in response.json()


class TestAnalyzeResponseShape:
    """POST /api/v1/analyze response includes all expected top-level keys."""

    _EXPECTED_KEYS: frozenset[str] = frozenset(
        {
            "language",
            "document_type",
            "word_stats",
            "sentence_stats",
            "repetition",
            "transitions",
            "burstiness",
            "readability",
            "cliches",
            "academic_risk",
            "highlights",
            "suggestions",
            "processing_time_ms",
        }
    )

    def setup_method(self):
        """Wire stub service for response shape inspection."""
        app.dependency_overrides[get_analysis_service] = lambda: _StubService()

    def teardown_method(self):
        """Restore real dependencies after each test."""
        app.dependency_overrides.clear()

    @pytest.mark.parametrize("key", sorted(_EXPECTED_KEYS))
    def test_top_level_key_present(self, key: str):
        """Each expected top-level key is present in the response."""
        response = _client.post("/api/v1/analyze", json=_VALID_BODY)
        assert key in response.json()

    def test_academic_risk_has_component_scores(self):
        """academic_risk contains a nested component_scores object."""
        response = _client.post("/api/v1/analyze", json=_VALID_BODY)
        assert "component_scores" in response.json()["academic_risk"]

    def test_academic_risk_has_overall_score(self):
        """academic_risk contains an overall_score float."""
        response = _client.post("/api/v1/analyze", json=_VALID_BODY)
        assert isinstance(response.json()["academic_risk"]["overall_score"], float)

    def test_academic_risk_has_explanations_list(self):
        """academic_risk contains an explanations list."""
        response = _client.post("/api/v1/analyze", json=_VALID_BODY)
        assert isinstance(response.json()["academic_risk"]["explanations"], list)
