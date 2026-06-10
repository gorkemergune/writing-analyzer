import pytest
from pydantic import ValidationError

from src.models.enums import DocumentType, Language, RiskLevel
from src.models.response import (
    AcademicRiskScore,
    AnalysisReport,
    BurstinessResult,
    ClicheResult,
    ComponentScores,
    Highlight,
    ReadabilityResult,
    RepetitionResult,
    RepeatedItem,
    SentenceStats,
    TransitionResult,
    WordStats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _component_scores(**overrides: float) -> ComponentScores:
    defaults = dict(
        repetition=40.0,
        transition_overuse=30.0,
        low_burstiness=50.0,
        lexical_poverty=25.0,
        cliche_density=15.0,
        readability=20.0,
    )
    return ComponentScores(**{**defaults, **overrides})


def _build_report() -> AnalysisReport:
    return AnalysisReport(
        language=Language.ENGLISH,
        document_type=DocumentType.ESSAY,
        word_stats=WordStats(
            total_words=150,
            unique_words=90,
            lexical_diversity=0.6,
            avg_word_length=5.2,
        ),
        sentence_stats=SentenceStats(
            total_sentences=10,
            avg_sentence_length=15.0,
            sentence_length_variance=12.5,
            min_sentence_length=5,
            max_sentence_length=30,
        ),
        repetition=RepetitionResult(
            repeated_words=[RepeatedItem(text="however", count=3, positions=[10, 45, 80])],
            repeated_phrases=[],
            repeated_openings=["The"],
            repetition_score=0.3,
        ),
        transitions=TransitionResult(
            found_transitions=["furthermore", "however", "therefore"],
            transition_density=0.4,
            overused_transitions=["furthermore"],
        ),
        burstiness=BurstinessResult(
            burstiness_score=0.2,
            sentence_variance=15.0,
            interpretation="Moderate burstiness",
        ),
        readability=ReadabilityResult(
            score=62.5,
            grade_level="Grade 10",
            interpretation="Standard",
        ),
        cliches=ClicheResult(
            found_cliches=["at the end of the day"],
            cliche_density=0.67,
        ),
        academic_risk=AcademicRiskScore(
            overall_score=58.0,
            risk_level=RiskLevel.HIGH,
            confidence=0.82,
            component_scores=_component_scores(),
        ),
        highlights=[
            Highlight(start=0, end=25, label="repeated_word", severity=RiskLevel.MODERATE),
        ],
        suggestions=["Vary sentence length more.", "Reduce transition word overuse."],
        processing_time_ms=142.7,
    )


# ---------------------------------------------------------------------------
# RepeatedItem
# ---------------------------------------------------------------------------

class TestRepeatedItem:
    def test_valid_construction(self):
        item = RepeatedItem(text="furthermore", count=3, positions=[0, 5, 12])
        assert item.text == "furthermore"
        assert item.count == 3
        assert item.positions == [0, 5, 12]

    def test_count_exactly_two_is_valid(self):
        item = RepeatedItem(text="word", count=2, positions=[0, 1])
        assert item.count == 2

    def test_count_one_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            RepeatedItem(text="word", count=1, positions=[0])
        assert any(e["loc"] == ("count",) for e in exc_info.value.errors())

    def test_count_zero_raises(self):
        with pytest.raises(ValidationError):
            RepeatedItem(text="word", count=0, positions=[])

    def test_count_negative_raises(self):
        with pytest.raises(ValidationError):
            RepeatedItem(text="word", count=-1, positions=[])

    def test_empty_positions_allowed(self):
        item = RepeatedItem(text="word", count=2, positions=[])
        assert item.positions == []

    def test_large_count_accepted(self):
        item = RepeatedItem(text="the", count=100, positions=list(range(100)))
        assert item.count == 100


# ---------------------------------------------------------------------------
# Highlight
# ---------------------------------------------------------------------------

class TestHighlight:
    def test_valid_construction(self):
        h = Highlight(start=10, end=20, label="cliche", severity=RiskLevel.HIGH)
        assert h.start == 10
        assert h.end == 20
        assert h.label == "cliche"
        assert h.severity is RiskLevel.HIGH

    def test_start_zero_is_valid(self):
        h = Highlight(start=0, end=5, label="repeated_phrase", severity=RiskLevel.LOW)
        assert h.start == 0

    def test_end_zero_is_valid(self):
        h = Highlight(start=0, end=0, label="test", severity=RiskLevel.LOW)
        assert h.end == 0

    def test_negative_start_raises(self):
        with pytest.raises(ValidationError):
            Highlight(start=-1, end=5, label="test", severity=RiskLevel.LOW)

    def test_negative_end_raises(self):
        with pytest.raises(ValidationError):
            Highlight(start=0, end=-1, label="test", severity=RiskLevel.LOW)

    def test_all_risk_levels_accepted(self):
        for level in RiskLevel:
            h = Highlight(start=0, end=10, label="test", severity=level)
            assert h.severity is level


# ---------------------------------------------------------------------------
# WordStats
# ---------------------------------------------------------------------------

class TestWordStats:
    def test_valid_construction(self):
        ws = WordStats(
            total_words=150,
            unique_words=90,
            lexical_diversity=0.6,
            avg_word_length=5.2,
        )
        assert ws.total_words == 150
        assert ws.unique_words == 90

    def test_lexical_diversity_at_zero(self):
        ws = WordStats(total_words=10, unique_words=1, lexical_diversity=0.0, avg_word_length=4.0)
        assert ws.lexical_diversity == 0.0

    def test_lexical_diversity_at_one(self):
        ws = WordStats(total_words=10, unique_words=10, lexical_diversity=1.0, avg_word_length=5.0)
        assert ws.lexical_diversity == 1.0

    def test_lexical_diversity_above_one_raises(self):
        with pytest.raises(ValidationError):
            WordStats(total_words=10, unique_words=5, lexical_diversity=1.01, avg_word_length=5.0)

    def test_lexical_diversity_negative_raises(self):
        with pytest.raises(ValidationError):
            WordStats(total_words=10, unique_words=5, lexical_diversity=-0.01, avg_word_length=5.0)

    def test_negative_total_words_raises(self):
        with pytest.raises(ValidationError):
            WordStats(total_words=-1, unique_words=0, lexical_diversity=0.5, avg_word_length=4.0)

    def test_negative_unique_words_raises(self):
        with pytest.raises(ValidationError):
            WordStats(total_words=10, unique_words=-1, lexical_diversity=0.5, avg_word_length=4.0)

    def test_negative_avg_word_length_raises(self):
        with pytest.raises(ValidationError):
            WordStats(total_words=10, unique_words=5, lexical_diversity=0.5, avg_word_length=-1.0)

    def test_all_zeros_valid(self):
        ws = WordStats(total_words=0, unique_words=0, lexical_diversity=0.0, avg_word_length=0.0)
        assert ws.total_words == 0


# ---------------------------------------------------------------------------
# SentenceStats
# ---------------------------------------------------------------------------

class TestSentenceStats:
    def test_valid_construction(self):
        ss = SentenceStats(
            total_sentences=10,
            avg_sentence_length=15.3,
            sentence_length_variance=12.5,
            min_sentence_length=5,
            max_sentence_length=30,
        )
        assert ss.total_sentences == 10
        assert ss.avg_sentence_length == 15.3

    def test_all_zeros_valid(self):
        ss = SentenceStats(
            total_sentences=0,
            avg_sentence_length=0.0,
            sentence_length_variance=0.0,
            min_sentence_length=0,
            max_sentence_length=0,
        )
        assert ss.total_sentences == 0

    def test_negative_total_sentences_raises(self):
        with pytest.raises(ValidationError):
            SentenceStats(
                total_sentences=-1,
                avg_sentence_length=10.0,
                sentence_length_variance=5.0,
                min_sentence_length=5,
                max_sentence_length=20,
            )

    def test_negative_avg_length_raises(self):
        with pytest.raises(ValidationError):
            SentenceStats(
                total_sentences=5,
                avg_sentence_length=-1.0,
                sentence_length_variance=5.0,
                min_sentence_length=5,
                max_sentence_length=20,
            )

    def test_negative_variance_raises(self):
        with pytest.raises(ValidationError):
            SentenceStats(
                total_sentences=5,
                avg_sentence_length=10.0,
                sentence_length_variance=-1.0,
                min_sentence_length=5,
                max_sentence_length=20,
            )

    def test_negative_min_length_raises(self):
        with pytest.raises(ValidationError):
            SentenceStats(
                total_sentences=5,
                avg_sentence_length=10.0,
                sentence_length_variance=5.0,
                min_sentence_length=-1,
                max_sentence_length=20,
            )

    def test_negative_max_length_raises(self):
        with pytest.raises(ValidationError):
            SentenceStats(
                total_sentences=5,
                avg_sentence_length=10.0,
                sentence_length_variance=5.0,
                min_sentence_length=5,
                max_sentence_length=-1,
            )


# ---------------------------------------------------------------------------
# RepetitionResult
# ---------------------------------------------------------------------------

class TestRepetitionResult:
    def test_valid_construction(self):
        result = RepetitionResult(
            repeated_words=[RepeatedItem(text="furthermore", count=3, positions=[0, 4, 8])],
            repeated_phrases=[],
            repeated_openings=["The"],
            repetition_score=0.35,
        )
        assert len(result.repeated_words) == 1
        assert result.repetition_score == 0.35

    def test_all_empty_lists_valid(self):
        result = RepetitionResult(
            repeated_words=[],
            repeated_phrases=[],
            repeated_openings=[],
            repetition_score=0.0,
        )
        assert result.repeated_words == []

    def test_repetition_score_at_zero(self):
        r = RepetitionResult(repeated_words=[], repeated_phrases=[], repeated_openings=[], repetition_score=0.0)
        assert r.repetition_score == 0.0

    def test_repetition_score_at_one(self):
        r = RepetitionResult(repeated_words=[], repeated_phrases=[], repeated_openings=[], repetition_score=1.0)
        assert r.repetition_score == 1.0

    def test_repetition_score_above_one_raises(self):
        with pytest.raises(ValidationError):
            RepetitionResult(repeated_words=[], repeated_phrases=[], repeated_openings=[], repetition_score=1.01)

    def test_repetition_score_negative_raises(self):
        with pytest.raises(ValidationError):
            RepetitionResult(repeated_words=[], repeated_phrases=[], repeated_openings=[], repetition_score=-0.1)


# ---------------------------------------------------------------------------
# BurstinessResult
# ---------------------------------------------------------------------------

class TestBurstinessResult:
    def test_valid_construction(self):
        r = BurstinessResult(burstiness_score=0.3, sentence_variance=25.4, interpretation="Moderate")
        assert r.burstiness_score == 0.3
        assert r.interpretation == "Moderate"

    def test_score_at_negative_one(self):
        r = BurstinessResult(burstiness_score=-1.0, sentence_variance=0.0, interpretation="Uniform")
        assert r.burstiness_score == -1.0

    def test_score_at_positive_one(self):
        r = BurstinessResult(burstiness_score=1.0, sentence_variance=100.0, interpretation="Very bursty")
        assert r.burstiness_score == 1.0

    def test_score_below_negative_one_raises(self):
        with pytest.raises(ValidationError):
            BurstinessResult(burstiness_score=-1.01, sentence_variance=0.0, interpretation="")

    def test_score_above_positive_one_raises(self):
        with pytest.raises(ValidationError):
            BurstinessResult(burstiness_score=1.01, sentence_variance=0.0, interpretation="")

    def test_negative_variance_raises(self):
        with pytest.raises(ValidationError):
            BurstinessResult(burstiness_score=0.0, sentence_variance=-0.1, interpretation="")

    def test_zero_variance_valid(self):
        r = BurstinessResult(burstiness_score=0.0, sentence_variance=0.0, interpretation="Flat")
        assert r.sentence_variance == 0.0


# ---------------------------------------------------------------------------
# ClicheResult
# ---------------------------------------------------------------------------

class TestClicheResult:
    def test_valid_construction(self):
        r = ClicheResult(found_cliches=["at the end of the day"], cliche_density=0.5)
        assert len(r.found_cliches) == 1
        assert r.cliche_density == 0.5

    def test_empty_cliches_valid(self):
        r = ClicheResult(found_cliches=[], cliche_density=0.0)
        assert r.found_cliches == []
        assert r.cliche_density == 0.0

    def test_negative_density_raises(self):
        with pytest.raises(ValidationError):
            ClicheResult(found_cliches=[], cliche_density=-0.1)


# ---------------------------------------------------------------------------
# ComponentScores
# ---------------------------------------------------------------------------

class TestComponentScores:
    def test_valid_construction(self):
        cs = _component_scores()
        assert cs.repetition == 40.0

    def test_all_zeros_valid(self):
        cs = ComponentScores(
            repetition=0.0, transition_overuse=0.0, low_burstiness=0.0,
            lexical_poverty=0.0, cliche_density=0.0, readability=0.0,
        )
        assert cs.repetition == 0.0

    def test_all_hundreds_valid(self):
        cs = ComponentScores(
            repetition=100.0, transition_overuse=100.0, low_burstiness=100.0,
            lexical_poverty=100.0, cliche_density=100.0, readability=100.0,
        )
        assert cs.repetition == 100.0

    def test_repetition_above_hundred_raises(self):
        with pytest.raises(ValidationError):
            _component_scores(repetition=100.1)

    def test_repetition_negative_raises(self):
        with pytest.raises(ValidationError):
            _component_scores(repetition=-0.1)

    def test_each_field_enforces_upper_bound(self):
        fields = ["transition_overuse", "low_burstiness", "lexical_poverty", "cliche_density", "readability"]
        for field in fields:
            with pytest.raises(ValidationError):
                _component_scores(**{field: 100.1})

    def test_each_field_enforces_lower_bound(self):
        fields = ["transition_overuse", "low_burstiness", "lexical_poverty", "cliche_density", "readability"]
        for field in fields:
            with pytest.raises(ValidationError):
                _component_scores(**{field: -0.1})


# ---------------------------------------------------------------------------
# AcademicRiskScore
# ---------------------------------------------------------------------------

class TestAcademicRiskScore:
    def test_valid_construction(self):
        score = AcademicRiskScore(
            overall_score=55.0,
            risk_level=RiskLevel.HIGH,
            confidence=0.85,
            component_scores=_component_scores(),
        )
        assert score.overall_score == 55.0
        assert score.risk_level is RiskLevel.HIGH

    def test_overall_score_at_zero(self):
        score = AcademicRiskScore(
            overall_score=0.0, risk_level=RiskLevel.LOW,
            confidence=1.0, component_scores=_component_scores(),
        )
        assert score.overall_score == 0.0

    def test_overall_score_at_hundred(self):
        score = AcademicRiskScore(
            overall_score=100.0, risk_level=RiskLevel.VERY_HIGH,
            confidence=0.9, component_scores=_component_scores(),
        )
        assert score.overall_score == 100.0

    def test_overall_score_above_hundred_raises(self):
        with pytest.raises(ValidationError):
            AcademicRiskScore(
                overall_score=100.1, risk_level=RiskLevel.VERY_HIGH,
                confidence=0.9, component_scores=_component_scores(),
            )

    def test_overall_score_negative_raises(self):
        with pytest.raises(ValidationError):
            AcademicRiskScore(
                overall_score=-0.1, risk_level=RiskLevel.LOW,
                confidence=0.9, component_scores=_component_scores(),
            )

    def test_confidence_at_zero(self):
        score = AcademicRiskScore(
            overall_score=50.0, risk_level=RiskLevel.MODERATE,
            confidence=0.0, component_scores=_component_scores(),
        )
        assert score.confidence == 0.0

    def test_confidence_at_one(self):
        score = AcademicRiskScore(
            overall_score=50.0, risk_level=RiskLevel.MODERATE,
            confidence=1.0, component_scores=_component_scores(),
        )
        assert score.confidence == 1.0

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValidationError):
            AcademicRiskScore(
                overall_score=50.0, risk_level=RiskLevel.MODERATE,
                confidence=1.01, component_scores=_component_scores(),
            )

    def test_confidence_negative_raises(self):
        with pytest.raises(ValidationError):
            AcademicRiskScore(
                overall_score=50.0, risk_level=RiskLevel.MODERATE,
                confidence=-0.01, component_scores=_component_scores(),
            )

    def test_all_risk_levels_accepted(self):
        for level in RiskLevel:
            score = AcademicRiskScore(
                overall_score=50.0, risk_level=level,
                confidence=0.8, component_scores=_component_scores(),
            )
            assert score.risk_level is level


# ---------------------------------------------------------------------------
# AnalysisReport
# ---------------------------------------------------------------------------

class TestAnalysisReport:
    def test_full_report_construction(self):
        report = _build_report()
        assert report.language is Language.ENGLISH
        assert report.document_type is DocumentType.ESSAY

    def test_word_stats_accessible(self):
        report = _build_report()
        assert report.word_stats.total_words == 150
        assert report.word_stats.lexical_diversity == 0.6

    def test_sentence_stats_accessible(self):
        report = _build_report()
        assert report.sentence_stats.total_sentences == 10

    def test_academic_risk_accessible(self):
        report = _build_report()
        assert report.academic_risk.overall_score == 58.0
        assert report.academic_risk.risk_level is RiskLevel.HIGH

    def test_component_scores_accessible(self):
        report = _build_report()
        assert report.academic_risk.component_scores.repetition == 40.0

    def test_highlights_list(self):
        report = _build_report()
        assert len(report.highlights) == 1
        assert report.highlights[0].label == "repeated_word"
        assert report.highlights[0].severity is RiskLevel.MODERATE

    def test_suggestions_list(self):
        report = _build_report()
        assert len(report.suggestions) == 2

    def test_processing_time_stored(self):
        report = _build_report()
        assert report.processing_time_ms == 142.7

    def test_negative_processing_time_raises(self):
        data = _build_report().model_dump()
        data["processing_time_ms"] = -1.0
        with pytest.raises(ValidationError):
            AnalysisReport(**data)

    def test_model_dump_has_all_top_level_keys(self):
        report = _build_report()
        data = report.model_dump()
        expected = {
            "language", "document_type", "word_stats", "sentence_stats",
            "repetition", "transitions", "burstiness", "readability",
            "cliches", "academic_risk", "highlights", "suggestions",
            "processing_time_ms",
        }
        assert set(data.keys()) == expected

    def test_json_round_trip(self):
        report = _build_report()
        restored = AnalysisReport.model_validate_json(report.model_dump_json())
        assert restored.language is report.language
        assert restored.academic_risk.overall_score == report.academic_risk.overall_score
        assert restored.word_stats.total_words == report.word_stats.total_words

    def test_turkish_report_construction(self):
        report = _build_report()
        data = report.model_dump()
        data["language"] = "tr"
        data["document_type"] = "academic"
        tr_report = AnalysisReport(**data)
        assert tr_report.language is Language.TURKISH
        assert tr_report.document_type is DocumentType.ACADEMIC

    def test_empty_highlights_valid(self):
        data = _build_report().model_dump()
        data["highlights"] = []
        report = AnalysisReport(**data)
        assert report.highlights == []

    def test_empty_suggestions_valid(self):
        data = _build_report().model_dump()
        data["suggestions"] = []
        report = AnalysisReport(**data)
        assert report.suggestions == []
