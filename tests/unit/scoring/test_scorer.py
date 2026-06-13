"""Comprehensive tests for AcademicRiskScorer and its helper functions."""

import pytest

from src.models.enums import RiskLevel
from src.models.response import (
    AcademicRiskScore,
    BurstinessResult,
    ClicheResult,
    ComponentScores,
    ReadabilityResult,
    RepetitionResult,
    SentenceStats,
    TransitionResult,
    WordStats,
)
from src.scoring.scorer import (
    AcademicRiskScorer,
    _build_components,
    _build_explanations,
    _classify_risk,
    _compute_confidence,
    _weighted_sum,
)
from src.scoring.weights import DEFAULT_WEIGHTS, ScoringWeights

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_word_stats(
    total_words: int = 300,
    unique_words: int = 200,
    lexical_diversity: float = 0.667,
    avg_word_length: float = 5.0,
) -> WordStats:
    return WordStats(
        total_words=total_words,
        unique_words=unique_words,
        lexical_diversity=lexical_diversity,
        avg_word_length=avg_word_length,
    )


def _make_sentence_stats(
    total_sentences: int = 20,
    avg_sentence_length: float = 15.0,
    sentence_length_variance: float = 20.0,
    min_sentence_length: int = 5,
    max_sentence_length: int = 30,
) -> SentenceStats:
    return SentenceStats(
        total_sentences=total_sentences,
        avg_sentence_length=avg_sentence_length,
        sentence_length_variance=sentence_length_variance,
        min_sentence_length=min_sentence_length,
        max_sentence_length=max_sentence_length,
    )


def _make_repetition(score: float = 0.0) -> RepetitionResult:
    return RepetitionResult(
        repeated_words=[],
        repeated_phrases=[],
        repeated_openings=[],
        repetition_score=score,
    )


def _make_transitions(score: float = 0.0) -> TransitionResult:
    return TransitionResult(
        transition_count=0,
        unique_transitions=[],
        repeated_transitions=[],
        transition_density=0.0,
        transition_score=score,
    )


def _make_burstiness(score: float = 0.5, value: float = 0.0) -> BurstinessResult:
    return BurstinessResult(
        burstiness_score=score,
        burstiness_value=value,
        classification="neutral",
    )


def _make_readability(score: float = 30.0) -> ReadabilityResult:
    return ReadabilityResult(
        readability_score=score,
        grade_level="College+",
        classification="difficult",
    )


def _make_cliches(score: float = 0.0, density: float = 0.0) -> ClicheResult:
    return ClicheResult(
        detected_cliches=[],
        cliche_count=0,
        cliche_density=density,
        cliche_score=score,
    )


def _make_zero_inputs() -> dict:
    return {
        "word_stats": _make_word_stats(total_words=0, unique_words=0, lexical_diversity=0.0),
        "sentence_stats": _make_sentence_stats(total_sentences=0),
        "repetition": _make_repetition(0.0),
        "transitions": _make_transitions(0.0),
        "burstiness": _make_burstiness(0.0, 0.0),
        "readability": _make_readability(0.0),
        "cliches": _make_cliches(0.0),
    }


# ---------------------------------------------------------------------------
# Profile helpers (arithmetic pre-verified)
# ---------------------------------------------------------------------------

# Profile A: LOW risk, human-like academic writing
# rep=0.15, tr=0.12, burst=0.35, lex_div=0.83, cliche=0.0, read=45
# components: rep=15, tr=12, burst=35, lex=17, cliche=0, read=8.33
# overall = 0.25×15 + 0.15×12 + 0.20×35 + 0.15×17 + 0.15×0 + 0.10×8.33
#         = 3.75 + 1.80 + 7.00 + 2.55 + 0.00 + 0.833 = 15.933 → LOW
_PROFILE_A = {
    "word_stats": _make_word_stats(total_words=400, lexical_diversity=0.83),
    "sentence_stats": _make_sentence_stats(),
    "repetition": _make_repetition(0.15),
    "transitions": _make_transitions(0.12),
    "burstiness": _make_burstiness(0.35),
    "readability": _make_readability(45.0),
    "cliches": _make_cliches(0.0),
}

# Profile B: MODERATE risk
# rep=0.40, tr=0.35, burst=0.50, lex_div=0.60, cliche=0.20, read=65
# components: rep=40, tr=35, burst=50, lex=40, cliche=20, read=41.67
# overall = 0.25×40 + 0.15×35 + 0.20×50 + 0.15×40 + 0.15×20 + 0.10×41.67
#         = 10.0 + 5.25 + 10.0 + 6.0 + 3.0 + 4.167 = 38.417 → MODERATE
_PROFILE_B = {
    "word_stats": _make_word_stats(total_words=300, lexical_diversity=0.60),
    "sentence_stats": _make_sentence_stats(),
    "repetition": _make_repetition(0.40),
    "transitions": _make_transitions(0.35),
    "burstiness": _make_burstiness(0.50),
    "readability": _make_readability(65.0),
    "cliches": _make_cliches(0.20),
}

# Profile C: HIGH risk
# rep=0.60, tr=0.55, burst=0.75, lex_div=0.35, cliche=0.50, read=78
# components: rep=60, tr=55, burst=75, lex=65, cliche=50, read=63.33
# overall = 0.25×60 + 0.15×55 + 0.20×75 + 0.15×65 + 0.15×50 + 0.10×63.33
#         = 15.0 + 8.25 + 15.0 + 9.75 + 7.5 + 6.333 = 61.833 → HIGH
_PROFILE_C = {
    "word_stats": _make_word_stats(total_words=300, lexical_diversity=0.35),
    "sentence_stats": _make_sentence_stats(),
    "repetition": _make_repetition(0.60),
    "transitions": _make_transitions(0.55),
    "burstiness": _make_burstiness(0.75),
    "readability": _make_readability(78.0),
    "cliches": _make_cliches(0.50),
}

# Profile D: VERY_HIGH risk
# rep=0.85, tr=0.80, burst=0.90, lex_div=0.20, cliche=0.80, read=88
# components: rep=85, tr=80, burst=90, lex=80, cliche=80, read=80
# overall = 0.25×85 + 0.15×80 + 0.20×90 + 0.15×80 + 0.15×80 + 0.10×80
#         = 21.25 + 12.0 + 18.0 + 12.0 + 12.0 + 8.0 = 83.25 → VERY_HIGH
_PROFILE_D = {
    "word_stats": _make_word_stats(total_words=300, lexical_diversity=0.20),
    "sentence_stats": _make_sentence_stats(),
    "repetition": _make_repetition(0.85),
    "transitions": _make_transitions(0.80),
    "burstiness": _make_burstiness(0.90),
    "readability": _make_readability(88.0),
    "cliches": _make_cliches(0.80),
}


# ---------------------------------------------------------------------------
# TestScoringWeights
# ---------------------------------------------------------------------------


class TestScoringWeightsDefaults:
    def test_default_weights_sum_to_one(self):
        w = ScoringWeights()
        total = (
            w.repetition
            + w.transition_overuse
            + w.low_burstiness
            + w.lexical_poverty
            + w.cliche_density
            + w.readability
        )
        assert abs(total - 1.0) < 0.001

    def test_default_repetition(self):
        assert DEFAULT_WEIGHTS.repetition == 0.25

    def test_default_transition_overuse(self):
        assert DEFAULT_WEIGHTS.transition_overuse == 0.15

    def test_default_low_burstiness(self):
        assert DEFAULT_WEIGHTS.low_burstiness == 0.20

    def test_default_lexical_poverty(self):
        assert DEFAULT_WEIGHTS.lexical_poverty == 0.15

    def test_default_cliche_density(self):
        assert DEFAULT_WEIGHTS.cliche_density == 0.15

    def test_default_readability(self):
        assert DEFAULT_WEIGHTS.readability == 0.10


class TestScoringWeightsValidation:
    def test_valid_custom_weights(self):
        w = ScoringWeights(
            repetition=0.30,
            transition_overuse=0.20,
            low_burstiness=0.20,
            lexical_poverty=0.10,
            cliche_density=0.10,
            readability=0.10,
        )
        assert w.repetition == 0.30

    def test_rejects_weights_summing_above_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            ScoringWeights(
                repetition=0.30,
                transition_overuse=0.20,
                low_burstiness=0.20,
                lexical_poverty=0.15,
                cliche_density=0.15,
                readability=0.15,  # sum = 1.15
            )

    def test_rejects_weights_summing_below_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            ScoringWeights(
                repetition=0.10,
                transition_overuse=0.10,
                low_burstiness=0.10,
                lexical_poverty=0.10,
                cliche_density=0.10,
                readability=0.10,  # sum = 0.60
            )

    def test_immutable_frozen_dataclass(self):
        from dataclasses import FrozenInstanceError

        w = ScoringWeights()
        with pytest.raises(FrozenInstanceError):
            w.repetition = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestComponentScores
# ---------------------------------------------------------------------------


class TestBuildComponents:
    def test_repetition_maps_score_times_100(self):
        components = _build_components(
            word_stats=_make_word_stats(lexical_diversity=1.0),
            repetition=_make_repetition(0.42),
            transitions=_make_transitions(0.0),
            burstiness=_make_burstiness(0.0),
            readability=_make_readability(0.0),
            cliches=_make_cliches(0.0),
        )
        assert components.repetition == pytest.approx(42.0)

    def test_transition_overuse_maps_score_times_100(self):
        components = _build_components(
            word_stats=_make_word_stats(lexical_diversity=1.0),
            repetition=_make_repetition(0.0),
            transitions=_make_transitions(0.67),
            burstiness=_make_burstiness(0.0),
            readability=_make_readability(0.0),
            cliches=_make_cliches(0.0),
        )
        assert components.transition_overuse == pytest.approx(67.0)

    def test_low_burstiness_maps_score_times_100(self):
        components = _build_components(
            word_stats=_make_word_stats(lexical_diversity=1.0),
            repetition=_make_repetition(0.0),
            transitions=_make_transitions(0.0),
            burstiness=_make_burstiness(0.80),
            readability=_make_readability(0.0),
            cliches=_make_cliches(0.0),
        )
        assert components.low_burstiness == pytest.approx(80.0)

    def test_lexical_poverty_is_one_minus_diversity(self):
        components = _build_components(
            word_stats=_make_word_stats(lexical_diversity=0.75),
            repetition=_make_repetition(0.0),
            transitions=_make_transitions(0.0),
            burstiness=_make_burstiness(0.0),
            readability=_make_readability(0.0),
            cliches=_make_cliches(0.0),
        )
        assert components.lexical_poverty == pytest.approx(25.0)

    def test_cliche_density_maps_score_times_100(self):
        components = _build_components(
            word_stats=_make_word_stats(lexical_diversity=1.0),
            repetition=_make_repetition(0.0),
            transitions=_make_transitions(0.0),
            burstiness=_make_burstiness(0.0),
            readability=_make_readability(0.0),
            cliches=_make_cliches(0.55),
        )
        assert components.cliche_density == pytest.approx(55.0)

    def test_readability_below_baseline_gives_zero(self):
        # readability_score <= 40.0 → component = 0
        components = _build_components(
            word_stats=_make_word_stats(lexical_diversity=1.0),
            repetition=_make_repetition(0.0),
            transitions=_make_transitions(0.0),
            burstiness=_make_burstiness(0.0),
            readability=_make_readability(40.0),
            cliches=_make_cliches(0.0),
        )
        assert components.readability == pytest.approx(0.0)

    def test_readability_at_zero_gives_zero(self):
        components = _build_components(
            word_stats=_make_word_stats(lexical_diversity=1.0),
            repetition=_make_repetition(0.0),
            transitions=_make_transitions(0.0),
            burstiness=_make_burstiness(0.0),
            readability=_make_readability(0.0),
            cliches=_make_cliches(0.0),
        )
        assert components.readability == pytest.approx(0.0)

    def test_readability_at_100_gives_100(self):
        # (100 - 40) / (100 - 40) * 100 = 100
        components = _build_components(
            word_stats=_make_word_stats(lexical_diversity=1.0),
            repetition=_make_repetition(0.0),
            transitions=_make_transitions(0.0),
            burstiness=_make_burstiness(0.0),
            readability=_make_readability(100.0),
            cliches=_make_cliches(0.0),
        )
        assert components.readability == pytest.approx(100.0)

    def test_readability_at_70_gives_50(self):
        # (70 - 40) / 60 * 100 = 50
        components = _build_components(
            word_stats=_make_word_stats(lexical_diversity=1.0),
            repetition=_make_repetition(0.0),
            transitions=_make_transitions(0.0),
            burstiness=_make_burstiness(0.0),
            readability=_make_readability(70.0),
            cliches=_make_cliches(0.0),
        )
        assert components.readability == pytest.approx(50.0)

    def test_readability_at_65_matches_profile_b(self):
        # (65 - 40) / 60 * 100 = 41.667
        components = _build_components(
            word_stats=_make_word_stats(lexical_diversity=1.0),
            repetition=_make_repetition(0.0),
            transitions=_make_transitions(0.0),
            burstiness=_make_burstiness(0.0),
            readability=_make_readability(65.0),
            cliches=_make_cliches(0.0),
        )
        assert components.readability == pytest.approx(41.667, abs=0.01)

    def test_all_zero_inputs_give_zero_components(self):
        components = _build_components(
            word_stats=_make_word_stats(lexical_diversity=1.0),
            repetition=_make_repetition(0.0),
            transitions=_make_transitions(0.0),
            burstiness=_make_burstiness(0.0),
            readability=_make_readability(0.0),
            cliches=_make_cliches(0.0),
        )
        assert components.repetition == 0.0
        assert components.transition_overuse == 0.0
        assert components.low_burstiness == 0.0
        assert components.cliche_density == 0.0
        assert components.readability == 0.0

    def test_full_diversity_gives_zero_lexical_poverty(self):
        components = _build_components(
            word_stats=_make_word_stats(lexical_diversity=1.0),
            repetition=_make_repetition(0.0),
            transitions=_make_transitions(0.0),
            burstiness=_make_burstiness(0.0),
            readability=_make_readability(0.0),
            cliches=_make_cliches(0.0),
        )
        assert components.lexical_poverty == pytest.approx(0.0)

    def test_zero_diversity_gives_100_lexical_poverty(self):
        components = _build_components(
            word_stats=_make_word_stats(lexical_diversity=0.0),
            repetition=_make_repetition(0.0),
            transitions=_make_transitions(0.0),
            burstiness=_make_burstiness(0.0),
            readability=_make_readability(0.0),
            cliches=_make_cliches(0.0),
        )
        assert components.lexical_poverty == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# TestWeightedSum
# ---------------------------------------------------------------------------


class TestWeightedSum:
    def test_all_zero_components_give_zero(self):
        components = ComponentScores(
            repetition=0.0,
            transition_overuse=0.0,
            low_burstiness=0.0,
            lexical_poverty=0.0,
            cliche_density=0.0,
            readability=0.0,
        )
        result = _weighted_sum(components, DEFAULT_WEIGHTS)
        assert result == pytest.approx(0.0)

    def test_all_100_components_give_100(self):
        components = ComponentScores(
            repetition=100.0,
            transition_overuse=100.0,
            low_burstiness=100.0,
            lexical_poverty=100.0,
            cliche_density=100.0,
            readability=100.0,
        )
        result = _weighted_sum(components, DEFAULT_WEIGHTS)
        assert result == pytest.approx(100.0)

    def test_single_repetition_component(self):
        # Only repetition=100, all others 0 → 0.25 × 100 = 25
        components = ComponentScores(
            repetition=100.0,
            transition_overuse=0.0,
            low_burstiness=0.0,
            lexical_poverty=0.0,
            cliche_density=0.0,
            readability=0.0,
        )
        result = _weighted_sum(components, DEFAULT_WEIGHTS)
        assert result == pytest.approx(25.0)

    def test_single_burstiness_component(self):
        # Only low_burstiness=100 → 0.20 × 100 = 20
        components = ComponentScores(
            repetition=0.0,
            transition_overuse=0.0,
            low_burstiness=100.0,
            lexical_poverty=0.0,
            cliche_density=0.0,
            readability=0.0,
        )
        result = _weighted_sum(components, DEFAULT_WEIGHTS)
        assert result == pytest.approx(20.0)

    def test_custom_weights_applied(self):
        custom = ScoringWeights(
            repetition=0.50,
            transition_overuse=0.10,
            low_burstiness=0.10,
            lexical_poverty=0.10,
            cliche_density=0.10,
            readability=0.10,
        )
        components = ComponentScores(
            repetition=100.0,
            transition_overuse=0.0,
            low_burstiness=0.0,
            lexical_poverty=0.0,
            cliche_density=0.0,
            readability=0.0,
        )
        result = _weighted_sum(components, custom)
        assert result == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# TestClassifyRisk
# ---------------------------------------------------------------------------


class TestClassifyRisk:
    def test_zero_is_low(self):
        assert _classify_risk(0.0) == RiskLevel.LOW

    def test_30_is_low(self):
        assert _classify_risk(30.0) == RiskLevel.LOW

    def test_30_001_is_moderate(self):
        assert _classify_risk(30.001) == RiskLevel.MODERATE

    def test_55_is_moderate(self):
        assert _classify_risk(55.0) == RiskLevel.MODERATE

    def test_55_001_is_high(self):
        assert _classify_risk(55.001) == RiskLevel.HIGH

    def test_75_is_high(self):
        assert _classify_risk(75.0) == RiskLevel.HIGH

    def test_75_001_is_very_high(self):
        assert _classify_risk(75.001) == RiskLevel.VERY_HIGH

    def test_100_is_very_high(self):
        assert _classify_risk(100.0) == RiskLevel.VERY_HIGH


# ---------------------------------------------------------------------------
# TestComputeConfidence
# ---------------------------------------------------------------------------


class TestComputeConfidence:
    def test_zero_words_gives_zero_confidence(self):
        assert _compute_confidence(0) == pytest.approx(0.0)

    def test_150_words_gives_half_confidence(self):
        assert _compute_confidence(150) == pytest.approx(0.5)

    def test_300_words_gives_full_confidence(self):
        assert _compute_confidence(300) == pytest.approx(1.0)

    def test_600_words_saturates_at_one(self):
        assert _compute_confidence(600) == pytest.approx(1.0)

    def test_1_word_gives_near_zero_confidence(self):
        assert _compute_confidence(1) == pytest.approx(1 / 300)

    def test_confidence_never_exceeds_one(self):
        for n in (0, 1, 100, 299, 300, 301, 1000):
            assert 0.0 <= _compute_confidence(n) <= 1.0


# ---------------------------------------------------------------------------
# TestBuildExplanations
# ---------------------------------------------------------------------------


class TestBuildExplanations:
    def _make_components(
        self,
        repetition: float = 0.0,
        transition_overuse: float = 0.0,
        low_burstiness: float = 0.0,
        lexical_poverty: float = 0.0,
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

    def test_all_low_components_return_empty_list(self):
        components = self._make_components()
        assert _build_explanations(components) == []

    def test_at_threshold_returns_empty(self):
        # Exactly at 40 is NOT above — should not trigger
        components = self._make_components(
            repetition=40.0,
            transition_overuse=40.0,
            low_burstiness=40.0,
            lexical_poverty=40.0,
            cliche_density=40.0,
            readability=40.0,
        )
        assert _build_explanations(components) == []

    def test_cliche_above_threshold_generates_explanation(self):
        components = self._make_components(cliche_density=50.0)
        entries = _build_explanations(components)
        assert len(entries) == 1
        assert "Clichés" in entries[0]
        assert "50" in entries[0]

    def test_lexical_poverty_above_threshold_generates_explanation(self):
        components = self._make_components(lexical_poverty=60.0)
        entries = _build_explanations(components)
        assert len(entries) == 1
        assert "Lexical" in entries[0]

    def test_low_burstiness_above_threshold_generates_explanation(self):
        components = self._make_components(low_burstiness=70.0)
        entries = _build_explanations(components)
        assert len(entries) == 1
        assert "uniform" in entries[0]

    def test_readability_above_threshold_generates_explanation(self):
        components = self._make_components(readability=55.0)
        entries = _build_explanations(components)
        assert len(entries) == 1
        assert "Readability" in entries[0]

    def test_repetition_above_threshold_generates_explanation(self):
        components = self._make_components(repetition=45.0)
        entries = _build_explanations(components)
        assert len(entries) == 1
        assert "Repetition" in entries[0]

    def test_transition_overuse_above_threshold_generates_explanation(self):
        components = self._make_components(transition_overuse=80.0)
        entries = _build_explanations(components)
        assert len(entries) == 1
        assert "Transition" in entries[0]

    def test_all_elevated_returns_six_explanations(self):
        components = self._make_components(
            repetition=50.0,
            transition_overuse=50.0,
            low_burstiness=50.0,
            lexical_poverty=50.0,
            cliche_density=50.0,
            readability=50.0,
        )
        entries = _build_explanations(components)
        assert len(entries) == 6

    def test_score_value_appears_in_explanation(self):
        components = self._make_components(cliche_density=75.0)
        entries = _build_explanations(components)
        assert "75" in entries[0]

    def test_explanations_use_probabilistic_language(self):
        components = self._make_components(
            repetition=60.0,
            low_burstiness=60.0,
            cliche_density=60.0,
        )
        for entry in _build_explanations(components):
            lowered = entry.lower()
            assert "ai-generated" not in lowered
            assert "definitely" not in lowered
            assert "certainly" not in lowered


# ---------------------------------------------------------------------------
# TestAcademicRiskScorerDefaults
# ---------------------------------------------------------------------------


class TestAcademicRiskScorerDefaults:
    def test_default_scorer_uses_default_weights(self):
        scorer = AcademicRiskScorer()
        assert scorer.weights == DEFAULT_WEIGHTS

    def test_weights_property_returns_active_weights(self):
        custom = ScoringWeights(
            repetition=0.30,
            transition_overuse=0.20,
            low_burstiness=0.20,
            lexical_poverty=0.10,
            cliche_density=0.10,
            readability=0.10,
        )
        scorer = AcademicRiskScorer(weights=custom)
        assert scorer.weights is custom

    def test_score_returns_academic_risk_score(self):
        scorer = AcademicRiskScorer()
        result = scorer.score(**_make_zero_inputs())
        assert isinstance(result, AcademicRiskScore)

    def test_zero_inputs_give_zero_overall_score(self):
        scorer = AcademicRiskScorer()
        result = scorer.score(**_make_zero_inputs())
        assert result.overall_score == pytest.approx(0.0, abs=0.01)

    def test_zero_inputs_give_low_risk(self):
        scorer = AcademicRiskScorer()
        result = scorer.score(**_make_zero_inputs())
        assert result.risk_level == RiskLevel.LOW

    def test_zero_inputs_give_zero_confidence(self):
        scorer = AcademicRiskScorer()
        result = scorer.score(**_make_zero_inputs())
        assert result.confidence == pytest.approx(0.0)

    def test_zero_inputs_give_empty_explanations(self):
        scorer = AcademicRiskScorer()
        result = scorer.score(**_make_zero_inputs())
        assert result.explanations == []

    def test_score_rounds_to_four_decimal_places(self):
        scorer = AcademicRiskScorer()
        result = scorer.score(**_PROFILE_B)
        # overall_score should have at most 4 decimal places
        rounded = round(result.overall_score, 4)
        assert result.overall_score == rounded


# ---------------------------------------------------------------------------
# TestProfileLow
# ---------------------------------------------------------------------------


class TestProfileLow:
    def test_overall_score(self):
        scorer = AcademicRiskScorer()
        result = scorer.score(**_PROFILE_A)
        assert result.overall_score == pytest.approx(15.933, abs=0.01)

    def test_risk_level(self):
        scorer = AcademicRiskScorer()
        result = scorer.score(**_PROFILE_A)
        assert result.risk_level == RiskLevel.LOW

    def test_confidence(self):
        scorer = AcademicRiskScorer()
        result = scorer.score(**_PROFILE_A)
        assert result.confidence == pytest.approx(1.0)

    def test_explanations_mostly_empty(self):
        scorer = AcademicRiskScorer()
        result = scorer.score(**_PROFILE_A)
        # Profile A has components: rep=15, tr=12, burst=35, lex=17, cliche=0, read=8.33
        # None exceed 40 → no explanations
        assert result.explanations == []


# ---------------------------------------------------------------------------
# TestProfileModerate
# ---------------------------------------------------------------------------


class TestProfileModerate:
    def test_overall_score(self):
        scorer = AcademicRiskScorer()
        result = scorer.score(**_PROFILE_B)
        assert result.overall_score == pytest.approx(38.417, abs=0.01)

    def test_risk_level(self):
        scorer = AcademicRiskScorer()
        result = scorer.score(**_PROFILE_B)
        assert result.risk_level == RiskLevel.MODERATE

    def test_some_explanations_generated(self):
        scorer = AcademicRiskScorer()
        result = scorer.score(**_PROFILE_B)
        # Profile B: rep=40 (≤40, no), tr=35 (no), burst=50 (yes), lex=40 (≤40, no),
        # cliche=20 (no), read=41.67 (yes)
        assert len(result.explanations) >= 1


# ---------------------------------------------------------------------------
# TestProfileHigh
# ---------------------------------------------------------------------------


class TestProfileHigh:
    def test_overall_score(self):
        scorer = AcademicRiskScorer()
        result = scorer.score(**_PROFILE_C)
        assert result.overall_score == pytest.approx(61.833, abs=0.01)

    def test_risk_level(self):
        scorer = AcademicRiskScorer()
        result = scorer.score(**_PROFILE_C)
        assert result.risk_level == RiskLevel.HIGH

    def test_multiple_explanations(self):
        scorer = AcademicRiskScorer()
        result = scorer.score(**_PROFILE_C)
        # Profile C: rep=60, tr=55, burst=75, lex=65, cliche=50, read=63.33 — all >40
        assert len(result.explanations) == 6


# ---------------------------------------------------------------------------
# TestProfileVeryHigh
# ---------------------------------------------------------------------------


class TestProfileVeryHigh:
    def test_overall_score(self):
        scorer = AcademicRiskScorer()
        result = scorer.score(**_PROFILE_D)
        assert result.overall_score == pytest.approx(83.25, abs=0.01)

    def test_risk_level(self):
        scorer = AcademicRiskScorer()
        result = scorer.score(**_PROFILE_D)
        assert result.risk_level == RiskLevel.VERY_HIGH

    def test_all_six_explanations(self):
        scorer = AcademicRiskScorer()
        result = scorer.score(**_PROFILE_D)
        assert len(result.explanations) == 6

    def test_confidence_at_300_words(self):
        scorer = AcademicRiskScorer()
        result = scorer.score(**_PROFILE_D)
        assert result.confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestRealisticEnglish
# ---------------------------------------------------------------------------


class TestRealisticEnglish:
    """Simulated analysis of English academic text with elevated risk signals."""

    def test_high_repetition_and_cliche_detected(self):
        # Simulate: repetitive academic essay with clichés and low burstiness
        scorer = AcademicRiskScorer()
        result = scorer.score(
            word_stats=_make_word_stats(total_words=350, lexical_diversity=0.45),
            sentence_stats=_make_sentence_stats(total_sentences=25),
            repetition=_make_repetition(0.55),
            transitions=_make_transitions(0.48),
            burstiness=_make_burstiness(0.70),
            readability=_make_readability(72.0),
            cliches=_make_cliches(0.40),
        )
        assert result.risk_level in (RiskLevel.HIGH, RiskLevel.VERY_HIGH)
        assert result.confidence >= 1.0

    def test_good_english_essay_scores_low(self):
        scorer = AcademicRiskScorer()
        result = scorer.score(
            word_stats=_make_word_stats(total_words=500, lexical_diversity=0.88),
            sentence_stats=_make_sentence_stats(total_sentences=30),
            repetition=_make_repetition(0.10),
            transitions=_make_transitions(0.08),
            burstiness=_make_burstiness(0.28),
            readability=_make_readability(38.0),
            cliches=_make_cliches(0.0),
        )
        assert result.risk_level == RiskLevel.LOW
        assert result.explanations == []


# ---------------------------------------------------------------------------
# TestRealisticTurkish
# ---------------------------------------------------------------------------


class TestRealisticTurkish:
    """Simulated analysis of Turkish academic text."""

    def test_formulaic_turkish_essay_scores_high(self):
        scorer = AcademicRiskScorer()
        result = scorer.score(
            word_stats=_make_word_stats(total_words=320, lexical_diversity=0.38),
            sentence_stats=_make_sentence_stats(total_sentences=20),
            repetition=_make_repetition(0.62),
            transitions=_make_transitions(0.58),
            burstiness=_make_burstiness(0.78),
            readability=_make_readability(75.0),
            cliches=_make_cliches(0.45),
        )
        assert result.risk_level in (RiskLevel.HIGH, RiskLevel.VERY_HIGH)

    def test_good_turkish_essay_scores_low_to_moderate(self):
        scorer = AcademicRiskScorer()
        result = scorer.score(
            word_stats=_make_word_stats(total_words=400, lexical_diversity=0.79),
            sentence_stats=_make_sentence_stats(total_sentences=25),
            repetition=_make_repetition(0.18),
            transitions=_make_transitions(0.14),
            burstiness=_make_burstiness(0.32),
            readability=_make_readability(42.0),
            cliches=_make_cliches(0.05),
        )
        assert result.risk_level in (RiskLevel.LOW, RiskLevel.MODERATE)


# ---------------------------------------------------------------------------
# TestInvariants (parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "profile,expected_level",
    [
        (_PROFILE_A, RiskLevel.LOW),
        (_PROFILE_B, RiskLevel.MODERATE),
        (_PROFILE_C, RiskLevel.HIGH),
        (_PROFILE_D, RiskLevel.VERY_HIGH),
    ],
)
class TestInvariants:
    def test_overall_score_in_range(self, profile, expected_level):
        scorer = AcademicRiskScorer()
        result = scorer.score(**profile)
        assert 0.0 <= result.overall_score <= 100.0

    def test_risk_level_matches_expected(self, profile, expected_level):
        scorer = AcademicRiskScorer()
        result = scorer.score(**profile)
        assert result.risk_level == expected_level

    def test_confidence_in_range(self, profile, expected_level):
        scorer = AcademicRiskScorer()
        result = scorer.score(**profile)
        assert 0.0 <= result.confidence <= 1.0

    def test_explanations_is_list(self, profile, expected_level):
        scorer = AcademicRiskScorer()
        result = scorer.score(**profile)
        assert isinstance(result.explanations, list)

    def test_component_scores_in_range(self, profile, expected_level):
        scorer = AcademicRiskScorer()
        result = scorer.score(**profile)
        cs = result.component_scores
        for value in (
            cs.repetition,
            cs.transition_overuse,
            cs.low_burstiness,
            cs.lexical_poverty,
            cs.cliche_density,
            cs.readability,
        ):
            assert 0.0 <= value <= 100.0


# ---------------------------------------------------------------------------
# TestCustomWeights
# ---------------------------------------------------------------------------


class TestCustomWeights:
    def test_custom_weights_change_overall_score(self):
        # Emphasise burstiness: if low_burstiness is the dominant weight,
        # a high burstiness score drives the total up.
        heavy_burst = ScoringWeights(
            repetition=0.10,
            transition_overuse=0.10,
            low_burstiness=0.50,
            lexical_poverty=0.10,
            cliche_density=0.10,
            readability=0.10,
        )
        scorer_default = AcademicRiskScorer()
        scorer_custom = AcademicRiskScorer(weights=heavy_burst)

        inputs = {
            "word_stats": _make_word_stats(lexical_diversity=0.50),
            "sentence_stats": _make_sentence_stats(),
            "repetition": _make_repetition(0.20),
            "transitions": _make_transitions(0.20),
            "burstiness": _make_burstiness(0.90),
            "readability": _make_readability(50.0),
            "cliches": _make_cliches(0.20),
        }
        result_default = scorer_default.score(**inputs)
        result_custom = scorer_custom.score(**inputs)
        assert result_custom.overall_score > result_default.overall_score

    def test_equal_weights_produce_equal_weighted_contribution(self):
        # When all weights are 1/6 ≈ 0.1667 each, each component contributes equally.
        # Use float-representable values to avoid rounding issues.
        equal = ScoringWeights(
            repetition=1 / 6,
            transition_overuse=1 / 6,
            low_burstiness=1 / 6,
            lexical_poverty=1 / 6,
            cliche_density=1 / 6,
            readability=1 / 6,
        )
        scorer = AcademicRiskScorer(weights=equal)
        result = scorer.score(
            word_stats=_make_word_stats(lexical_diversity=0.40),  # lex_poverty=60
            sentence_stats=_make_sentence_stats(),
            repetition=_make_repetition(0.60),  # rep=60
            transitions=_make_transitions(0.60),  # tr=60
            burstiness=_make_burstiness(0.60),  # burst=60
            readability=_make_readability(76.0),  # read=(76-40)/60*100=60
            cliches=_make_cliches(0.60),  # cliche=60
        )
        assert result.overall_score == pytest.approx(60.0, abs=0.1)
