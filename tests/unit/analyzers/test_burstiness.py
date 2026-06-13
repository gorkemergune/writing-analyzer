"""Unit tests for BurstinessAnalyzer."""

import pytest

from src.analyzers.burstiness import (
    BurstinessAnalyzer,
    _classify,
    _compute_burstiness,
)
from src.models.analysis import AnalysisContext
from src.models.enums import DocumentType, Language
from src.models.response import BurstinessResult

# ---------------------------------------------------------------------------
# Module-level AnalysisContext fixtures with pre-verified expected values.
#
# B = (σ − μ) / (σ + μ),  score = (1 − B) / 2
# All σ values are population standard deviations.
# ---------------------------------------------------------------------------

# ── _EMPTY ───────────────────────────────────────────────────────────────────
# No tokens, no sentences → neutral default (insufficient data).
_EMPTY = AnalysisContext(
    raw_text="",
    language=Language.ENGLISH,
    document_type=DocumentType.ESSAY,
    cleaned_text="",
    tokens=(),
    sentences=(),
    stems=(),
    sentence_token_counts=(),
)

# ── _SINGLE_SENTENCE ─────────────────────────────────────────────────────────
# One sentence of 15 tokens → neutral default (can't measure variability).
_SINGLE_SENTENCE = AnalysisContext(
    raw_text="stub",
    language=Language.ENGLISH,
    document_type=DocumentType.ESSAY,
    cleaned_text="stub",
    tokens=("word",) * 15,
    sentences=("stub sentence",),
    stems=("stem",) * 15,
    sentence_token_counts=(15,),
)

# ── _VERY_UNIFORM ─────────────────────────────────────────────────────────────
# Five sentences, all exactly 10 tokens.
# μ = 10, σ = 0  →  B = (0−10)/(0+10) = −1.0
# score = (1 − (−1.0)) / 2 = 1.0
# classification = very_uniform
_VERY_UNIFORM = AnalysisContext(
    raw_text="stub",
    language=Language.ENGLISH,
    document_type=DocumentType.ESSAY,
    cleaned_text="stub",
    tokens=("word",) * 50,
    sentences=("stub",) * 5,
    stems=("stem",) * 50,
    sentence_token_counts=(10, 10, 10, 10, 10),
)

# ── _UNIFORM ──────────────────────────────────────────────────────────────────
# Four sentences: [5, 10, 15, 20].
# μ = 12.5
# σ² = (7.5² + 2.5² + 2.5² + 7.5²) / 4 = 125/4 = 31.25  →  σ ≈ 5.5902
# B = (5.5902 − 12.5) / (5.5902 + 12.5) = −6.9098 / 18.0902 ≈ −0.3820
# score ≈ (1 − (−0.3820)) / 2 ≈ 0.6910
# classification = uniform
_UNIFORM = AnalysisContext(
    raw_text="stub",
    language=Language.ENGLISH,
    document_type=DocumentType.ESSAY,
    cleaned_text="stub",
    tokens=("word",) * 50,
    sentences=("stub",) * 4,
    stems=("stem",) * 50,
    sentence_token_counts=(5, 10, 15, 20),
)

# ── _NEUTRAL ──────────────────────────────────────────────────────────────────
# Four sentences: [1, 4, 8, 20].
# μ = 33/4 = 8.25
# σ² = (7.25² + 4.25² + 0.25² + 11.75²) / 4
#     = (52.5625 + 18.0625 + 0.0625 + 138.0625) / 4 = 208.75 / 4 = 52.1875
# σ ≈ 7.2239
# B = (7.2239 − 8.25) / (7.2239 + 8.25) = −1.0261 / 15.4739 ≈ −0.0663
# score ≈ (1 − (−0.0663)) / 2 ≈ 0.5332
# classification = neutral
_NEUTRAL = AnalysisContext(
    raw_text="stub",
    language=Language.ENGLISH,
    document_type=DocumentType.ESSAY,
    cleaned_text="stub",
    tokens=("word",) * 33,
    sentences=("stub",) * 4,
    stems=("stem",) * 33,
    sentence_token_counts=(1, 4, 8, 20),
)

# ── _BURSTY ───────────────────────────────────────────────────────────────────
# Five sentences: [1, 1, 1, 1, 60].
# μ = 64/5 = 12.8
# σ² = (4 × 11.8² + 47.2²) / 5 = (4 × 139.24 + 2227.84) / 5 = 556.96
# σ ≈ 23.6002
# B = (23.6002 − 12.8) / (23.6002 + 12.8) = 10.8002 / 36.4002 ≈ 0.2967
# score ≈ (1 − 0.2967) / 2 ≈ 0.3517
# classification = bursty
_BURSTY = AnalysisContext(
    raw_text="stub",
    language=Language.ENGLISH,
    document_type=DocumentType.ESSAY,
    cleaned_text="stub",
    tokens=("word",) * 64,
    sentences=("stub",) * 5,
    stems=("stem",) * 64,
    sentence_token_counts=(1, 1, 1, 1, 60),
)

# ── _HIGHLY_BURSTY ────────────────────────────────────────────────────────────
# 20 sentences: nineteen 1-token sentences + one 200-token sentence.
# μ = 219/20 = 10.95
# σ² = (19 × 9.95² + 189.05²) / 20 = (19 × 99.0025 + 35729.9025) / 20
#     = 37610.95 / 20 = 1880.5475  →  σ ≈ 43.3654
# B = (43.3654 − 10.95) / (43.3654 + 10.95) = 32.4154 / 54.3154 ≈ 0.5969
# score ≈ (1 − 0.5969) / 2 ≈ 0.2016
# classification = highly_bursty
_HIGHLY_BURSTY = AnalysisContext(
    raw_text="stub",
    language=Language.ENGLISH,
    document_type=DocumentType.ESSAY,
    cleaned_text="stub",
    tokens=("word",) * 219,
    sentences=("stub",) * 20,
    stems=("stem",) * 219,
    sentence_token_counts=(1,) * 19 + (200,),
)

# ── _TR_UNIFORM ───────────────────────────────────────────────────────────────
# Turkish academic text: six sentences with moderate, uniform lengths.
# sentence_token_counts = (5, 12, 8, 18, 10, 15)
# μ = 68/6 ≈ 11.3333
# deviations: −6.333, +0.667, −3.333, +6.667, −1.333, +3.667
# σ² = (40.111 + 0.444 + 11.111 + 44.444 + 1.778 + 13.444) / 6
#     ≈ 111.332 / 6 ≈ 18.5553  →  σ ≈ 4.3077
# B = (4.3077 − 11.3333) / (4.3077 + 11.3333) = −7.0256 / 15.6410 ≈ −0.4492
# score ≈ (1 − (−0.4492)) / 2 ≈ 0.7246
# classification = uniform
_TR_UNIFORM = AnalysisContext(
    raw_text="stub",
    language=Language.TURKISH,
    document_type=DocumentType.ESSAY,
    cleaned_text="stub",
    tokens=("kelime",) * 68,
    sentences=("stub",) * 6,
    stems=("kok",) * 68,
    sentence_token_counts=(5, 12, 8, 18, 10, 15),
)

# ── _TR_BURSTY ────────────────────────────────────────────────────────────────
# Turkish creative text: highly uneven sentence lengths.
# sentence_token_counts = (2, 1, 25, 3, 22, 2)
# μ = 55/6 ≈ 9.1667
# deviations: −7.167, −8.167, +15.833, −6.167, +12.833, −7.167
# σ² = (51.366 + 66.699 + 250.685 + 38.032 + 164.685 + 51.366) / 6
#     ≈ 622.833 / 6 ≈ 103.8056  →  σ ≈ 10.1885
# B = (10.1885 − 9.1667) / (10.1885 + 9.1667) = 1.0218 / 19.3552 ≈ 0.0528
# score ≈ (1 − 0.0528) / 2 ≈ 0.4736
# classification = neutral
_TR_NEUTRAL = AnalysisContext(
    raw_text="stub",
    language=Language.TURKISH,
    document_type=DocumentType.ESSAY,
    cleaned_text="stub",
    tokens=("kelime",) * 55,
    sentences=("stub",) * 6,
    stems=("kok",) * 55,
    sentence_token_counts=(2, 1, 25, 3, 22, 2),
)

# ---------------------------------------------------------------------------
# All fixtures bundled for parametrized invariant tests.
# ---------------------------------------------------------------------------

_ALL_FIXTURES = [
    _EMPTY,
    _SINGLE_SENTENCE,
    _VERY_UNIFORM,
    _UNIFORM,
    _NEUTRAL,
    _BURSTY,
    _HIGHLY_BURSTY,
    _TR_UNIFORM,
    _TR_NEUTRAL,
]

_VALID_CLASSIFICATIONS = frozenset(
    {"very_uniform", "uniform", "neutral", "bursty", "highly_bursty"}
)


# ===========================================================================
# Test classes
# ===========================================================================


class TestAnalyzerIdentity:
    """Verify static properties of the analyzer object."""

    def test_name(self) -> None:
        assert BurstinessAnalyzer().name == "burstiness"

    def test_analyze_returns_burstiness_result(self) -> None:
        result = BurstinessAnalyzer().analyze(_EMPTY)
        assert isinstance(result, BurstinessResult)

    def test_two_distinct_instances_are_independent(self) -> None:
        a = BurstinessAnalyzer()
        b = BurstinessAnalyzer()
        assert a.analyze(_VERY_UNIFORM) == b.analyze(_VERY_UNIFORM)


class TestEmptyInput:
    """No sentences → neutral default."""

    def setup_method(self) -> None:
        self._result = BurstinessAnalyzer().analyze(_EMPTY)

    def test_burstiness_value_is_zero(self) -> None:
        assert self._result.burstiness_value == 0.0

    def test_burstiness_score_is_neutral(self) -> None:
        assert self._result.burstiness_score == pytest.approx(0.5)

    def test_classification_is_neutral(self) -> None:
        assert self._result.classification == "neutral"


class TestSingleSentence:
    """One sentence → neutral default (no intra-document rhythm)."""

    def setup_method(self) -> None:
        self._result = BurstinessAnalyzer().analyze(_SINGLE_SENTENCE)

    def test_burstiness_value_is_zero(self) -> None:
        assert self._result.burstiness_value == 0.0

    def test_burstiness_score_is_neutral(self) -> None:
        assert self._result.burstiness_score == pytest.approx(0.5)

    def test_classification_is_neutral(self) -> None:
        assert self._result.classification == "neutral"


class TestVeryUniform:
    """All sentences identical length → B = −1.0 (maximum uniformity)."""

    def setup_method(self) -> None:
        self._result = BurstinessAnalyzer().analyze(_VERY_UNIFORM)

    def test_burstiness_value(self) -> None:
        assert self._result.burstiness_value == pytest.approx(-1.0)

    def test_burstiness_score(self) -> None:
        assert self._result.burstiness_score == pytest.approx(1.0)

    def test_classification(self) -> None:
        assert self._result.classification == "very_uniform"


class TestUniform:
    """Sentences [5,10,15,20]: moderate uniformity, B ≈ −0.382."""

    def setup_method(self) -> None:
        self._result = BurstinessAnalyzer().analyze(_UNIFORM)

    def test_burstiness_value(self) -> None:
        assert self._result.burstiness_value == pytest.approx(-0.38197, abs=1e-4)

    def test_burstiness_score(self) -> None:
        assert self._result.burstiness_score == pytest.approx(0.69099, abs=1e-4)

    def test_classification(self) -> None:
        assert self._result.classification == "uniform"


class TestNeutral:
    """Sentences [1,4,8,20]: near-neutral, B ≈ −0.066."""

    def setup_method(self) -> None:
        self._result = BurstinessAnalyzer().analyze(_NEUTRAL)

    def test_burstiness_value(self) -> None:
        assert self._result.burstiness_value == pytest.approx(-0.0663, abs=1e-3)

    def test_burstiness_score(self) -> None:
        assert self._result.burstiness_score == pytest.approx(0.5332, abs=1e-3)

    def test_classification(self) -> None:
        assert self._result.classification == "neutral"


class TestBursty:
    """Sentences [1,1,1,1,60]: high variation, B ≈ 0.297."""

    def setup_method(self) -> None:
        self._result = BurstinessAnalyzer().analyze(_BURSTY)

    def test_burstiness_value(self) -> None:
        assert self._result.burstiness_value == pytest.approx(0.2967, abs=1e-3)

    def test_burstiness_score(self) -> None:
        assert self._result.burstiness_score == pytest.approx(0.3517, abs=1e-3)

    def test_classification(self) -> None:
        assert self._result.classification == "bursty"


class TestHighlyBursty:
    """19×(1) + 1×(200): extreme outlier, B ≈ 0.597."""

    def setup_method(self) -> None:
        self._result = BurstinessAnalyzer().analyze(_HIGHLY_BURSTY)

    def test_burstiness_value(self) -> None:
        assert self._result.burstiness_value == pytest.approx(0.5969, abs=1e-3)

    def test_burstiness_score(self) -> None:
        assert self._result.burstiness_score == pytest.approx(0.2016, abs=1e-3)

    def test_classification(self) -> None:
        assert self._result.classification == "highly_bursty"


class TestTurkishUniform:
    """Turkish academic prose with uniform sentence lengths."""

    def setup_method(self) -> None:
        self._result = BurstinessAnalyzer().analyze(_TR_UNIFORM)

    def test_burstiness_value(self) -> None:
        assert self._result.burstiness_value == pytest.approx(-0.4492, abs=1e-3)

    def test_burstiness_score(self) -> None:
        assert self._result.burstiness_score == pytest.approx(0.7246, abs=1e-3)

    def test_classification(self) -> None:
        assert self._result.classification == "uniform"

    def test_language_does_not_affect_result(self) -> None:
        # BurstinessAnalyzer is language-agnostic; same sentence_token_counts
        # in an English context must produce the same B value.
        en_ctx = AnalysisContext(
            raw_text="stub",
            language=Language.ENGLISH,
            document_type=DocumentType.ESSAY,
            cleaned_text="stub",
            tokens=("word",) * 68,
            sentences=("stub",) * 6,
            stems=("stem",) * 68,
            sentence_token_counts=(5, 12, 8, 18, 10, 15),
        )
        en_result = BurstinessAnalyzer().analyze(en_ctx)
        assert en_result.burstiness_value == pytest.approx(
            self._result.burstiness_value, abs=1e-9
        )


class TestTurkishNeutral:
    """Turkish text with mixed short and long sentences → neutral range."""

    def setup_method(self) -> None:
        self._result = BurstinessAnalyzer().analyze(_TR_NEUTRAL)

    def test_burstiness_value(self) -> None:
        assert self._result.burstiness_value == pytest.approx(0.0528, abs=1e-3)

    def test_burstiness_score(self) -> None:
        assert self._result.burstiness_score == pytest.approx(0.4736, abs=1e-3)

    def test_classification(self) -> None:
        assert self._result.classification == "neutral"


class TestScoreInvariants:
    """Parametrized invariants that must hold for every fixture."""

    @pytest.mark.parametrize("ctx", _ALL_FIXTURES)
    def test_burstiness_value_in_range(self, ctx: AnalysisContext) -> None:
        result = BurstinessAnalyzer().analyze(ctx)
        assert -1.0 <= result.burstiness_value <= 1.0

    @pytest.mark.parametrize("ctx", _ALL_FIXTURES)
    def test_burstiness_score_in_range(self, ctx: AnalysisContext) -> None:
        result = BurstinessAnalyzer().analyze(ctx)
        assert 0.0 <= result.burstiness_score <= 1.0

    @pytest.mark.parametrize("ctx", _ALL_FIXTURES)
    def test_classification_is_valid_label(self, ctx: AnalysisContext) -> None:
        result = BurstinessAnalyzer().analyze(ctx)
        assert result.classification in _VALID_CLASSIFICATIONS

    @pytest.mark.parametrize("ctx", _ALL_FIXTURES)
    def test_score_derived_from_value(self, ctx: AnalysisContext) -> None:
        result = BurstinessAnalyzer().analyze(ctx)
        expected_score = (1.0 - result.burstiness_value) / 2.0
        assert result.burstiness_score == pytest.approx(expected_score, abs=1e-9)

    @pytest.mark.parametrize("ctx", _ALL_FIXTURES)
    def test_deterministic(self, ctx: AnalysisContext) -> None:
        analyzer = BurstinessAnalyzer()
        assert analyzer.analyze(ctx) == analyzer.analyze(ctx)

    @pytest.mark.parametrize("ctx", _ALL_FIXTURES)
    def test_context_not_mutated(self, ctx: AnalysisContext) -> None:
        original_counts = ctx.sentence_token_counts
        BurstinessAnalyzer().analyze(ctx)
        assert ctx.sentence_token_counts == original_counts


class TestClassificationBoundaries:
    """Exact threshold boundary values for _classify."""

    def test_just_below_minus_0_6_is_very_uniform(self) -> None:
        assert _classify(-0.601) == "very_uniform"

    def test_exactly_minus_0_6_is_uniform(self) -> None:
        assert _classify(-0.6) == "uniform"

    def test_just_above_minus_0_6_is_uniform(self) -> None:
        assert _classify(-0.599) == "uniform"

    def test_just_below_minus_0_2_is_uniform(self) -> None:
        assert _classify(-0.201) == "uniform"

    def test_exactly_minus_0_2_is_neutral(self) -> None:
        assert _classify(-0.2) == "neutral"

    def test_just_above_minus_0_2_is_neutral(self) -> None:
        assert _classify(-0.199) == "neutral"

    def test_just_below_0_2_is_neutral(self) -> None:
        assert _classify(0.199) == "neutral"

    def test_exactly_0_2_is_bursty(self) -> None:
        assert _classify(0.2) == "bursty"

    def test_just_above_0_2_is_bursty(self) -> None:
        assert _classify(0.201) == "bursty"

    def test_just_below_0_5_is_bursty(self) -> None:
        assert _classify(0.499) == "bursty"

    def test_exactly_0_5_is_highly_bursty(self) -> None:
        assert _classify(0.5) == "highly_bursty"

    def test_above_0_5_is_highly_bursty(self) -> None:
        assert _classify(0.8) == "highly_bursty"

    def test_minus_1_is_very_uniform(self) -> None:
        assert _classify(-1.0) == "very_uniform"

    def test_0_is_neutral(self) -> None:
        assert _classify(0.0) == "neutral"

    def test_1_is_highly_bursty(self) -> None:
        assert _classify(1.0) == "highly_bursty"


class TestComputeBurstiness:
    """Unit tests for _compute_burstiness helper."""

    def test_all_same_returns_minus_one(self) -> None:
        assert _compute_burstiness((5, 5, 5, 5)) == pytest.approx(-1.0)

    def test_zero_denom_returns_zero(self) -> None:
        # All sentences have zero tokens: μ=0, σ=0 → denom=0.
        assert _compute_burstiness((0, 0, 0)) == 0.0

    def test_known_value_uniform(self) -> None:
        # [5, 10, 15, 20] → B ≈ −0.38197
        assert _compute_burstiness((5, 10, 15, 20)) == pytest.approx(-0.38197, abs=1e-4)

    def test_known_value_bursty(self) -> None:
        # [1, 1, 1, 1, 60] → B ≈ 0.2967
        assert _compute_burstiness((1, 1, 1, 1, 60)) == pytest.approx(0.2967, abs=1e-3)

    def test_two_equal_sentences_returns_minus_one(self) -> None:
        assert _compute_burstiness((8, 8)) == pytest.approx(-1.0)

    def test_result_bounded_above_minus_one(self) -> None:
        assert _compute_burstiness((1, 100)) >= -1.0

    def test_result_bounded_above_one(self) -> None:
        result = _compute_burstiness((1,) * 19 + (200,))
        assert result <= 1.0

    def test_uses_population_not_sample_stddev(self) -> None:
        # For [10, 10, 10, 10], population σ=0 → B exactly −1.
        # Sample σ would also be 0, so distinguish with [1, 3]:
        # population σ = 1.0, sample σ = sqrt(2) ≈ 1.4142.
        # μ = 2.0
        # population: B = (1.0 − 2.0) / (1.0 + 2.0) = −1/3 ≈ −0.3333
        # sample:     B = (1.4142 − 2.0) / (1.4142 + 2.0) = −0.5858/3.4142 ≈ −0.1716
        result = _compute_burstiness((1, 3))
        assert result == pytest.approx(-1 / 3, abs=1e-9)


class TestRealisticEnglish:
    """Realistic English academic text scenarios."""

    def test_formulaic_ai_like_essay(self) -> None:
        # Essay with near-identical sentence lengths: AI-like uniformity.
        # Counts: [18, 19, 18, 20, 18, 19] — small spread around 18–20 words.
        # μ = 112/6 ≈ 18.667, σ small → B well below −0.6 → very_uniform.
        ctx = AnalysisContext(
            raw_text="stub",
            language=Language.ENGLISH,
            document_type=DocumentType.ESSAY,
            cleaned_text="stub",
            tokens=("word",) * 112,
            sentences=("stub",) * 6,
            stems=("stem",) * 112,
            sentence_token_counts=(18, 19, 18, 20, 18, 19),
        )
        result = BurstinessAnalyzer().analyze(ctx)
        assert result.classification == "very_uniform"
        assert result.burstiness_score > 0.9

    def test_human_like_varied_prose(self) -> None:
        # Academic prose with deliberate variation: short topic sentences,
        # long explanatory sentences, brief transitional sentences.
        # Counts: [4, 22, 3, 24, 5, 20, 4]
        ctx = AnalysisContext(
            raw_text="stub",
            language=Language.ENGLISH,
            document_type=DocumentType.ESSAY,
            cleaned_text="stub",
            tokens=("word",) * 82,
            sentences=("stub",) * 7,
            stems=("stem",) * 82,
            sentence_token_counts=(4, 22, 3, 24, 5, 20, 4),
        )
        result = BurstinessAnalyzer().analyze(ctx)
        assert result.burstiness_value > -0.2
        assert result.burstiness_score < 0.6

    def test_research_paper_moderate_variation(self) -> None:
        # Research paper: longer sentences on average, moderate variation.
        # Counts: [20, 25, 15, 30, 22, 18]
        ctx = AnalysisContext(
            raw_text="stub",
            language=Language.ENGLISH,
            document_type=DocumentType.RESEARCH,
            cleaned_text="stub",
            tokens=("word",) * 130,
            sentences=("stub",) * 6,
            stems=("stem",) * 130,
            sentence_token_counts=(20, 25, 15, 30, 22, 18),
        )
        result = BurstinessAnalyzer().analyze(ctx)
        assert result.classification in {"uniform", "neutral"}

    def test_high_score_signals_uniform_risk(self) -> None:
        # A very uniform text should produce a high burstiness_score (risk).
        ctx = AnalysisContext(
            raw_text="stub",
            language=Language.ENGLISH,
            document_type=DocumentType.ESSAY,
            cleaned_text="stub",
            tokens=("word",) * 60,
            sentences=("stub",) * 6,
            stems=("stem",) * 60,
            sentence_token_counts=(10, 10, 10, 10, 10, 10),
        )
        result = BurstinessAnalyzer().analyze(ctx)
        assert result.burstiness_score == pytest.approx(1.0)

    def test_low_score_signals_varied_writing(self) -> None:
        # A highly bursty text should produce a low burstiness_score (low risk).
        result = BurstinessAnalyzer().analyze(_HIGHLY_BURSTY)
        assert result.burstiness_score < 0.25


class TestRealisticTurkish:
    """Realistic Turkish academic text scenarios."""

    def test_turkish_uniform_essay(self) -> None:
        # Typical formulaic Turkish essay: consistent sentence lengths.
        result = BurstinessAnalyzer().analyze(_TR_UNIFORM)
        assert result.classification == "uniform"
        assert result.burstiness_value < -0.2

    def test_turkish_neutral_mixed_prose(self) -> None:
        result = BurstinessAnalyzer().analyze(_TR_NEUTRAL)
        assert result.classification == "neutral"
        assert -0.2 <= result.burstiness_value < 0.2

    def test_turkish_very_uniform(self) -> None:
        # All sentences same length → maximum risk signal regardless of language.
        ctx = AnalysisContext(
            raw_text="stub",
            language=Language.TURKISH,
            document_type=DocumentType.ESSAY,
            cleaned_text="stub",
            tokens=("kelime",) * 60,
            sentences=("stub",) * 6,
            stems=("kok",) * 60,
            sentence_token_counts=(10, 10, 10, 10, 10, 10),
        )
        result = BurstinessAnalyzer().analyze(ctx)
        assert result.classification == "very_uniform"
        assert result.burstiness_score == pytest.approx(1.0)

    def test_language_agnostic(self) -> None:
        # The same sentence_token_counts must produce identical results
        # regardless of the Language field in the context.
        counts = (6, 14, 9, 20, 7, 18)
        en_ctx = AnalysisContext(
            raw_text="stub",
            language=Language.ENGLISH,
            document_type=DocumentType.ESSAY,
            cleaned_text="stub",
            tokens=("word",) * sum(counts),
            sentences=("stub",) * len(counts),
            stems=("stem",) * sum(counts),
            sentence_token_counts=counts,
        )
        tr_ctx = AnalysisContext(
            raw_text="stub",
            language=Language.TURKISH,
            document_type=DocumentType.ESSAY,
            cleaned_text="stub",
            tokens=("kelime",) * sum(counts),
            sentences=("stub",) * len(counts),
            stems=("kok",) * sum(counts),
            sentence_token_counts=counts,
        )
        en_result = BurstinessAnalyzer().analyze(en_ctx)
        tr_result = BurstinessAnalyzer().analyze(tr_ctx)
        assert en_result.burstiness_value == pytest.approx(tr_result.burstiness_value)
        assert en_result.classification == tr_result.classification
