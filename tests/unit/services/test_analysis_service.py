"""Unit tests for AnalysisService and its pure helper functions."""

from typing import Any

import pytest

from src.analyzers.base import BaseAnalyzer
from src.models.analysis import AnalysisContext
from src.models.enums import DocumentType, Language, RiskLevel
from src.models.request import AnalysisRequest
from src.models.response import (
    AcademicRiskScore,
    AnalysisReport,
    BurstinessResult,
    ClicheResult,
    ComponentScores,
    ReadabilityResult,
    RepeatedItem,
    RepetitionResult,
    SentenceStats,
    TransitionResult,
    WordStats,
)
from src.scoring.scorer import AcademicRiskScorer
from src.scoring.weights import ScoringWeights
from src.services.analysis import (
    AnalysisService,
    _build_highlights,
    _build_suggestions,
    _find_spans,
    _score_to_risk,
)
from src.services.language_detector import LanguageDetector
from src.services.tokenizer import TokenizerService

# ---------------------------------------------------------------------------
# Shared stub helpers
# ---------------------------------------------------------------------------


class _StubAnalyzer(BaseAnalyzer):  # type: ignore[type-arg]
    """Analyzer stub that returns a pre-set result for any context."""

    def __init__(self, analyzer_name: str, result: Any) -> None:
        self._analyzer_name = analyzer_name
        self._result = result

    @property
    def name(self) -> str:
        return self._analyzer_name

    def analyze(self, _context: AnalysisContext) -> Any:  # type: ignore[override]
        return self._result


class _StubTokenizer(TokenizerService):
    """Tokenizer stub that bypasses NLTK setup and returns a fixed context."""

    def __init__(self) -> None:
        object.__init__(self)  # skip TokenizerService.__init__ intentionally

    def build_context(
        self,
        raw_text: str,
        language: Language,
        document_type: DocumentType,
    ) -> AnalysisContext:
        return AnalysisContext(
            raw_text=raw_text,
            language=language,
            document_type=document_type,
            cleaned_text=raw_text,
            tokens=("word", "another", "word", "test", "academic"),
            sentences=("word another word test academic.",),
            stems=("word", "anoth", "word", "test", "academ"),
            sentence_token_counts=(5,),
        )


def _make_word_stats(total_words: int = 100, diversity: float = 0.7) -> WordStats:
    return WordStats(
        total_words=total_words,
        unique_words=int(total_words * diversity),
        lexical_diversity=diversity,
        avg_word_length=6.0,
    )


def _make_sentence_stats() -> SentenceStats:
    return SentenceStats(
        total_sentences=5,
        avg_sentence_length=15.0,
        sentence_length_variance=10.0,
        min_sentence_length=8,
        max_sentence_length=22,
    )


def _make_repetition(score: float = 0.1) -> RepetitionResult:
    return RepetitionResult(
        repeated_words=[],
        repeated_phrases=[],
        repeated_openings=[],
        repetition_score=score,
    )


def _make_transitions(score: float = 0.1) -> TransitionResult:
    return TransitionResult(
        transition_count=1,
        unique_transitions=["furthermore"],
        repeated_transitions=[],
        transition_density=0.2,
        transition_score=score,
    )


def _make_burstiness(score: float = 0.4) -> BurstinessResult:
    return BurstinessResult(
        burstiness_score=score,
        burstiness_value=0.2,
        classification="neutral",
    )


def _make_readability(score: float = 35.0) -> ReadabilityResult:
    return ReadabilityResult(
        readability_score=score,
        grade_level="College+",
        classification="difficult",
    )


def _make_cliches(score: float = 0.0) -> ClicheResult:
    return ClicheResult(
        detected_cliches=[],
        cliche_count=0,
        cliche_density=0.0,
        cliche_score=score,
    )


def _make_components(
    repetition: float = 10.0,
    transition_overuse: float = 10.0,
    low_burstiness: float = 40.0,
    lexical_poverty: float = 30.0,
    cliche_density: float = 0.0,
    readability: float = 0.0,
) -> ComponentScores:
    return ComponentScores(
        repetition=repetition,
        transition_overuse=transition_overuse,
        low_burstiness=low_burstiness,
        lexical_poverty=lexical_poverty,
        cliche_density=cliche_density,
        readability=readability,
    )


def _make_academic_risk(
    overall: float = 20.0,
    risk_level: RiskLevel = RiskLevel.LOW,
    components: ComponentScores | None = None,
) -> AcademicRiskScore:
    if components is None:
        components = _make_components()
    return AcademicRiskScore(
        overall_score=overall,
        risk_level=risk_level,
        confidence=1.0,
        component_scores=components,
        explanations=[],
    )


def _make_stub_service(
    language: Language = Language.ENGLISH,
    repetition_score: float = 0.1,
    transition_score: float = 0.1,
    burstiness_score: float = 0.4,
    readability_score: float = 35.0,
    cliche_score: float = 0.0,
    lexical_diversity: float = 0.7,
    total_words: int = 100,
) -> AnalysisService:
    """Build a fully-stubbed AnalysisService for orchestration tests."""
    word_stats = _make_word_stats(total_words, lexical_diversity)
    return AnalysisService(
        language_detector=LanguageDetector(_detect_fn=lambda _t: language.value),
        tokenizer=_StubTokenizer(),
        word_stats_analyzer=_StubAnalyzer("word_stats", word_stats),
        sentence_stats_analyzer=_StubAnalyzer("sentence_stats", _make_sentence_stats()),
        repetition_analyzer=_StubAnalyzer("repetition", _make_repetition(repetition_score)),
        transition_analyzer=_StubAnalyzer("transition", _make_transitions(transition_score)),
        burstiness_analyzer=_StubAnalyzer("burstiness", _make_burstiness(burstiness_score)),
        readability_analyzer=_StubAnalyzer("readability", _make_readability(readability_score)),
        cliche_analyzer=_StubAnalyzer("cliche", _make_cliches(cliche_score)),
    )


# ---------------------------------------------------------------------------
# TestFindSpans
# ---------------------------------------------------------------------------


class TestFindSpans:
    def test_single_match(self):
        assert _find_spans("In conclusion, text.", "in conclusion") == [(0, 12)]

    def test_two_matches(self):
        spans = _find_spans("In conclusion, in conclusion.", "in conclusion")
        assert len(spans) == 2
        assert spans[0] == (0, 12)
        assert spans[1] == (15, 27)

    def test_case_insensitive_upper(self):
        spans = _find_spans("IN CONCLUSION.", "in conclusion")
        assert spans == [(0, 12)]

    def test_case_insensitive_mixed(self):
        spans = _find_spans("Furthermore, furthermore.", "furthermore")
        assert len(spans) == 2

    def test_no_match_returns_empty(self):
        assert _find_spans("Nothing here.", "in conclusion") == []

    def test_turkish_phrase_found(self):
        text = "Sonuç olarak, sonuç olarak tekrar."
        spans = _find_spans(text, "sonuç olarak")
        assert len(spans) == 2

    def test_offsets_are_correct(self):
        text = "some text in conclusion more text"
        spans = _find_spans(text, "in conclusion")
        assert len(spans) == 1
        start, end = spans[0]
        assert text[start:end].lower() == "in conclusion"

    def test_phrase_with_special_regex_chars(self):
        # "the fact of the matter is" contains no special chars — just sanity
        text = "the fact of the matter is clear"
        spans = _find_spans(text, "the fact of the matter is")
        assert len(spans) == 1

    def test_non_overlapping_matches(self):
        # "aa" in "aaaa" yields 2 non-overlapping matches: (0,2) and (2,4)
        spans = _find_spans("aaaa", "aa")
        assert spans == [(0, 2), (2, 4)]

    def test_phrase_longer_than_text_returns_empty(self):
        assert _find_spans("short", "a very long phrase indeed") == []

    def test_empty_text_returns_empty(self):
        assert _find_spans("", "in conclusion") == []

    def test_empty_phrase_handled(self):
        # re.escape("") matches every position — just verify it does not crash
        _find_spans("some text", "")


# ---------------------------------------------------------------------------
# TestScoreToRisk
# ---------------------------------------------------------------------------


class TestScoreToRisk:
    def test_zero_is_low(self):
        assert _score_to_risk(0.0) == RiskLevel.LOW

    def test_30_is_low(self):
        assert _score_to_risk(30.0) == RiskLevel.LOW

    def test_31_is_moderate(self):
        assert _score_to_risk(31.0) == RiskLevel.MODERATE

    def test_55_is_moderate(self):
        assert _score_to_risk(55.0) == RiskLevel.MODERATE

    def test_56_is_high(self):
        assert _score_to_risk(56.0) == RiskLevel.HIGH

    def test_75_is_high(self):
        assert _score_to_risk(75.0) == RiskLevel.HIGH

    def test_76_is_very_high(self):
        assert _score_to_risk(76.0) == RiskLevel.VERY_HIGH

    def test_100_is_very_high(self):
        assert _score_to_risk(100.0) == RiskLevel.VERY_HIGH


# ---------------------------------------------------------------------------
# TestBuildSuggestions
# ---------------------------------------------------------------------------


class TestBuildSuggestions:
    def test_all_low_returns_empty(self):
        risk = _make_academic_risk(components=_make_components(
            repetition=10.0, transition_overuse=10.0, low_burstiness=10.0,
            lexical_poverty=10.0, cliche_density=0.0, readability=0.0,
        ))
        assert _build_suggestions(risk) == []

    def test_at_threshold_returns_empty(self):
        # Exactly at 40 is NOT above — no suggestion
        risk = _make_academic_risk(components=_make_components(
            repetition=40.0, transition_overuse=40.0, low_burstiness=40.0,
            lexical_poverty=40.0, cliche_density=40.0, readability=40.0,
        ))
        assert _build_suggestions(risk) == []

    def test_cliche_above_threshold_adds_suggestion(self):
        risk = _make_academic_risk(components=_make_components(cliche_density=50.0))
        suggestions = _build_suggestions(risk)
        assert len(suggestions) == 1
        assert "in conclusion" in suggestions[0]

    def test_lexical_poverty_adds_suggestion(self):
        risk = _make_academic_risk(components=_make_components(lexical_poverty=60.0))
        suggestions = _build_suggestions(risk)
        assert any("vocabulary" in s for s in suggestions)

    def test_low_burstiness_adds_suggestion(self):
        risk = _make_academic_risk(components=_make_components(low_burstiness=55.0))
        suggestions = _build_suggestions(risk)
        assert any("sentence length" in s.lower() for s in suggestions)

    def test_readability_adds_suggestion(self):
        risk = _make_academic_risk(components=_make_components(readability=60.0))
        suggestions = _build_suggestions(risk)
        assert any("complexity" in s.lower() for s in suggestions)

    def test_repetition_adds_suggestion(self):
        risk = _make_academic_risk(components=_make_components(repetition=50.0))
        suggestions = _build_suggestions(risk)
        assert any("repetition" in s.lower() for s in suggestions)

    def test_transition_overuse_adds_suggestion(self):
        risk = _make_academic_risk(components=_make_components(transition_overuse=45.0))
        suggestions = _build_suggestions(risk)
        assert any("transition" in s.lower() for s in suggestions)

    def test_all_elevated_returns_six_suggestions(self):
        risk = _make_academic_risk(components=_make_components(
            repetition=50.0, transition_overuse=50.0, low_burstiness=50.0,
            lexical_poverty=50.0, cliche_density=50.0, readability=50.0,
        ))
        assert len(_build_suggestions(risk)) == 6

    def test_suggestions_use_probabilistic_language(self):
        risk = _make_academic_risk(components=_make_components(
            repetition=60.0, cliche_density=60.0,
        ))
        for s in _build_suggestions(risk):
            assert "ai-generated" not in s.lower()
            assert "definitely" not in s.lower()


# ---------------------------------------------------------------------------
# TestBuildHighlights helpers (module-level for reuse across tests)
# ---------------------------------------------------------------------------


def _make_cliche_result(phrases: list[str]) -> ClicheResult:
    return ClicheResult(
        detected_cliches=phrases,
        cliche_count=len(phrases),
        cliche_density=float(len(phrases)),
        cliche_score=min(1.0, len(phrases) / 5.0),
    )


def _make_transition_result(repeated: list[str]) -> TransitionResult:
    return TransitionResult(
        transition_count=len(repeated) * 2,
        unique_transitions=repeated,
        repeated_transitions=repeated,
        transition_density=1.0,
        transition_score=0.6,
    )


def _make_repetition_result(phrases: list[tuple[str, int]]) -> RepetitionResult:
    items = [
        RepeatedItem(text=p, count=c, positions=list(range(c)))
        for p, c in phrases
    ]
    return RepetitionResult(
        repeated_words=[],
        repeated_phrases=items,
        repeated_openings=[],
        repetition_score=0.3,
    )


# ---------------------------------------------------------------------------
# TestBuildHighlights
# ---------------------------------------------------------------------------


class TestBuildHighlights:
    """Tests for _build_highlights using in-memory result objects."""

    def test_no_matches_returns_empty(self):
        text = "A clean academic text with no formulaic phrases."
        risk = _make_academic_risk()
        highlights = _build_highlights(
            text,
            _make_repetition_result([]),
            _make_transition_result([]),
            _make_cliche_result([]),
            risk,
        )
        assert highlights == []

    def test_cliche_match_generates_highlight(self):
        text = "In conclusion, this work demonstrates the result."
        risk = _make_academic_risk(components=_make_components(cliche_density=60.0))
        highlights = _build_highlights(
            text,
            _make_repetition_result([]),
            _make_transition_result([]),
            _make_cliche_result(["in conclusion"]),
            risk,
        )
        assert len(highlights) == 1
        assert highlights[0].label == "cliche"
        assert text[highlights[0].start : highlights[0].end].lower() == "in conclusion"

    def test_transition_match_generates_highlight(self):
        text = "Furthermore, the data shows. Furthermore, evidence confirms."
        risk = _make_academic_risk()
        highlights = _build_highlights(
            text,
            _make_repetition_result([]),
            _make_transition_result(["furthermore"]),
            _make_cliche_result([]),
            risk,
        )
        labels = [h.label for h in highlights]
        assert all(label == "transition_overuse" for label in labels)
        assert len(highlights) == 2

    def test_repeated_phrase_generates_highlight(self):
        text = "digital tools enhance learning. digital tools improve outcomes."
        risk = _make_academic_risk()
        highlights = _build_highlights(
            text,
            _make_repetition_result([("digital tools", 2)]),
            _make_transition_result([]),
            _make_cliche_result([]),
            risk,
        )
        assert any(h.label == "repeated_phrase" for h in highlights)

    def test_cliche_beats_transition_on_overlap(self):
        # "in conclusion" is both a cliché and a transition — cliché wins.
        text = "In conclusion, the study shows results."
        risk = _make_academic_risk(components=_make_components(cliche_density=60.0))
        highlights = _build_highlights(
            text,
            _make_repetition_result([]),
            _make_transition_result(["in conclusion"]),
            _make_cliche_result(["in conclusion"]),
            risk,
        )
        cliche_labels = [h.label for h in highlights if h.label == "cliche"]
        assert len(cliche_labels) == 1
        # No duplicate span with transition_overuse label
        tr_labels = [h.label for h in highlights if h.label == "transition_overuse"]
        assert len(tr_labels) == 0

    def test_highlights_in_document_order(self):
        text = "Furthermore, it is important to note that the study is complete."
        risk = _make_academic_risk(components=_make_components(cliche_density=60.0))
        highlights = _build_highlights(
            text,
            _make_repetition_result([]),
            _make_transition_result(["furthermore"]),
            _make_cliche_result(["it is important to note that"]),
            risk,
        )
        starts = [h.start for h in highlights]
        assert starts == sorted(starts)

    def test_highlights_do_not_overlap(self):
        text = "In conclusion, in conclusion, the work is done."
        risk = _make_academic_risk(components=_make_components(cliche_density=60.0))
        highlights = _build_highlights(
            text,
            _make_repetition_result([]),
            _make_transition_result([]),
            _make_cliche_result(["in conclusion"]),
            risk,
        )
        for i in range(len(highlights) - 1):
            assert highlights[i].end <= highlights[i + 1].start

    def test_severity_comes_from_component_score(self):
        text = "In conclusion, results are shown."
        risk = _make_academic_risk(
            components=_make_components(cliche_density=80.0)
        )
        highlights = _build_highlights(
            text,
            _make_repetition_result([]),
            _make_transition_result([]),
            _make_cliche_result(["in conclusion"]),
            risk,
        )
        assert len(highlights) == 1
        assert highlights[0].severity == RiskLevel.VERY_HIGH

    def test_empty_text_returns_empty(self):
        risk = _make_academic_risk()
        highlights = _build_highlights(
            "",
            _make_repetition_result([]),
            _make_transition_result([]),
            _make_cliche_result([]),
            risk,
        )
        assert highlights == []

    def test_phrase_highlights_capped_at_five(self):
        # Six different repeated phrases — only first five are highlighted.
        phrases = [
            ("alpha beta", 2), ("gamma delta", 2), ("epsilon zeta", 2),
            ("eta theta", 2), ("iota kappa", 2), ("lambda mu", 2),
        ]
        text = " ".join(p for p, _ in phrases) + " " + " ".join(p for p, _ in phrases)
        risk = _make_academic_risk()
        highlights = _build_highlights(
            text,
            _make_repetition_result(phrases),
            _make_transition_result([]),
            _make_cliche_result([]),
            risk,
        )
        rep_highlights = [h for h in highlights if h.label == "repeated_phrase"]
        # At most 5 phrases are searched (2 occurrences each = up to 10 spans)
        unique_spans = {(h.start, h.end) for h in rep_highlights}
        assert len(unique_spans) <= 10


# ---------------------------------------------------------------------------
# TestLanguageResolution
# ---------------------------------------------------------------------------


class TestLanguageResolution:
    def test_english_language_from_request_used(self):
        svc = _make_stub_service(language=Language.ENGLISH)
        req = AnalysisRequest(
            text="A" * 60,
            document_type=DocumentType.ESSAY,
            language=Language.ENGLISH,
        )
        report = svc.analyze(req)
        assert report.language == Language.ENGLISH

    def test_turkish_language_from_request_used(self):
        svc = _make_stub_service(language=Language.TURKISH)
        req = AnalysisRequest(
            text="A" * 60,
            document_type=DocumentType.ESSAY,
            language=Language.TURKISH,
        )
        report = svc.analyze(req)
        assert report.language == Language.TURKISH

    def test_auto_detect_called_when_language_is_none(self):
        detected: list[str] = []

        def _capture(text: str) -> str:
            detected.append(text)
            return "en"

        svc = AnalysisService(
            language_detector=LanguageDetector(_detect_fn=_capture),
            tokenizer=_StubTokenizer(),
            word_stats_analyzer=_StubAnalyzer("word_stats", _make_word_stats()),
            sentence_stats_analyzer=_StubAnalyzer("sentence_stats", _make_sentence_stats()),
            repetition_analyzer=_StubAnalyzer("repetition", _make_repetition()),
            transition_analyzer=_StubAnalyzer("transition", _make_transitions()),
            burstiness_analyzer=_StubAnalyzer("burstiness", _make_burstiness()),
            readability_analyzer=_StubAnalyzer("readability", _make_readability()),
            cliche_analyzer=_StubAnalyzer("cliche", _make_cliches()),
        )
        req = AnalysisRequest(text="A" * 60, document_type=DocumentType.ESSAY)
        svc.analyze(req)
        assert len(detected) == 1

    def test_detect_not_called_when_language_provided(self):
        called: list[bool] = []

        def _should_not_call(text: str) -> str:
            called.append(True)
            return "en"

        svc = AnalysisService(
            language_detector=LanguageDetector(_detect_fn=_should_not_call),
            tokenizer=_StubTokenizer(),
            word_stats_analyzer=_StubAnalyzer("word_stats", _make_word_stats()),
            sentence_stats_analyzer=_StubAnalyzer("sentence_stats", _make_sentence_stats()),
            repetition_analyzer=_StubAnalyzer("repetition", _make_repetition()),
            transition_analyzer=_StubAnalyzer("transition", _make_transitions()),
            burstiness_analyzer=_StubAnalyzer("burstiness", _make_burstiness()),
            readability_analyzer=_StubAnalyzer("readability", _make_readability()),
            cliche_analyzer=_StubAnalyzer("cliche", _make_cliches()),
        )
        req = AnalysisRequest(
            text="A" * 60,
            document_type=DocumentType.ESSAY,
            language=Language.ENGLISH,
        )
        svc.analyze(req)
        assert called == []


# ---------------------------------------------------------------------------
# TestReportShape
# ---------------------------------------------------------------------------


class TestReportShape:
    @pytest.fixture
    def report(self) -> AnalysisReport:
        svc = _make_stub_service()
        return svc.analyze(
            AnalysisRequest(
                text="A" * 60,
                document_type=DocumentType.ESSAY,
                language=Language.ENGLISH,
            )
        )

    def test_returns_analysis_report(self, report: AnalysisReport):
        assert isinstance(report, AnalysisReport)

    def test_language_matches_request(self, report: AnalysisReport):
        assert report.language == Language.ENGLISH

    def test_document_type_matches_request(self, report: AnalysisReport):
        assert report.document_type == DocumentType.ESSAY

    def test_processing_time_is_nonnegative(self, report: AnalysisReport):
        assert report.processing_time_ms >= 0.0

    def test_overall_score_in_range(self, report: AnalysisReport):
        assert 0.0 <= report.academic_risk.overall_score <= 100.0

    def test_confidence_in_range(self, report: AnalysisReport):
        assert 0.0 <= report.academic_risk.confidence <= 1.0

    def test_risk_level_is_valid(self, report: AnalysisReport):
        assert report.academic_risk.risk_level in list(RiskLevel)

    def test_explanations_is_list(self, report: AnalysisReport):
        assert isinstance(report.academic_risk.explanations, list)

    def test_highlights_is_list(self, report: AnalysisReport):
        assert isinstance(report.highlights, list)

    def test_suggestions_is_list(self, report: AnalysisReport):
        assert isinstance(report.suggestions, list)

    def test_word_stats_total_words_nonnegative(self, report: AnalysisReport):
        assert report.word_stats.total_words >= 0

    def test_sentence_stats_total_sentences_nonnegative(self, report: AnalysisReport):
        assert report.sentence_stats.total_sentences >= 0


# ---------------------------------------------------------------------------
# TestPipelineOrchestration
# ---------------------------------------------------------------------------


class TestPipelineOrchestration:
    def test_stub_analyzer_output_reaches_scorer(self):
        # If repetition_score=0.6 → repetition component = 60
        svc = _make_stub_service(repetition_score=0.6)
        req = AnalysisRequest(
            text="A" * 60,
            document_type=DocumentType.ESSAY,
            language=Language.ENGLISH,
        )
        report = svc.analyze(req)
        assert report.academic_risk.component_scores.repetition == pytest.approx(60.0)

    def test_stub_cliche_output_reaches_scorer(self):
        svc = _make_stub_service(cliche_score=0.8)
        req = AnalysisRequest(
            text="A" * 60,
            document_type=DocumentType.ESSAY,
            language=Language.ENGLISH,
        )
        report = svc.analyze(req)
        assert report.academic_risk.component_scores.cliche_density == pytest.approx(80.0)

    def test_lexical_diversity_reaches_scorer(self):
        svc = _make_stub_service(lexical_diversity=0.5)
        req = AnalysisRequest(
            text="A" * 60,
            document_type=DocumentType.ESSAY,
            language=Language.ENGLISH,
        )
        report = svc.analyze(req)
        # lexical_poverty = (1 - 0.5) * 100 = 50
        assert report.academic_risk.component_scores.lexical_poverty == pytest.approx(50.0)

    def test_custom_scorer_weights_applied(self):
        # Give repetition 50% weight; with stub rep_score=1.0, component=100
        heavy_rep = ScoringWeights(
            repetition=0.50,
            transition_overuse=0.10,
            low_burstiness=0.10,
            lexical_poverty=0.10,
            cliche_density=0.10,
            readability=0.10,
        )
        svc = _make_stub_service(
            repetition_score=1.0,
            transition_score=0.0,
            burstiness_score=0.0,
            readability_score=0.0,
            cliche_score=0.0,
            lexical_diversity=1.0,
        )
        svc._scorer = AcademicRiskScorer(weights=heavy_rep)  # type: ignore[attr-defined]
        req = AnalysisRequest(
            text="A" * 60,
            document_type=DocumentType.ESSAY,
            language=Language.ENGLISH,
        )
        report = svc.analyze(req)
        # With lexical_diversity=1.0 → lexical_poverty=0, readability=35→component=0
        # overall ≈ 0.50 * 100 + rest ≈ 50.0
        assert report.academic_risk.overall_score == pytest.approx(50.0, abs=1.0)

    def test_word_stats_passed_to_confidence(self):
        # 300-word text → confidence = 1.0; 150-word text → 0.5
        svc_full = _make_stub_service(total_words=300)
        svc_half = _make_stub_service(total_words=150)
        req = AnalysisRequest(
            text="A" * 60,
            document_type=DocumentType.ESSAY,
            language=Language.ENGLISH,
        )
        assert svc_full.analyze(req).academic_risk.confidence == pytest.approx(1.0)
        assert svc_half.analyze(req).academic_risk.confidence == pytest.approx(0.5)
