"""Unit tests for SentenceStatisticsAnalyzer."""

import pytest

from src.analyzers.sentence_statistics import SentenceStatisticsAnalyzer
from src.models.analysis import AnalysisContext
from src.models.enums import DocumentType, Language
from src.models.response import SentenceStats

# ---------------------------------------------------------------------------
# Module-level minimal contexts for exact-value assertions.
# SentenceStatisticsAnalyzer reads only sentence_token_counts, so tokens,
# sentences, and stems can be minimal stubs.
# ---------------------------------------------------------------------------


def _ctx(counts: tuple[int, ...]) -> AnalysisContext:
    """Build a minimal AnalysisContext with the given per-sentence token counts."""
    total = sum(counts)
    return AnalysisContext(
        raw_text="stub",
        language=Language.ENGLISH,
        document_type=DocumentType.ESSAY,
        cleaned_text="stub",
        tokens=tuple("w" for _ in range(total)),
        sentences=tuple(f"S{i}." for i in range(len(counts))),
        stems=tuple("w" for _ in range(total)),
        sentence_token_counts=counts,
    )


# Contexts used across multiple test classes.
# Expected values verified by hand:

# Empty — no sentences
_EMPTY = _ctx(())

# Single sentence, 5 tokens
# total=1, avg=5.0, variance=0.0, min=5, max=5
_SINGLE = _ctx((5,))

# Two sentences with equal length
# total=2, avg=8.0, variance=0.0, min=8, max=8
_TWO_EQUAL = _ctx((8, 8))

# Three sentences with different lengths
# mean=21/3=7.0, variance=((4-7)²+(10-7)²+(7-7)²)/3=(9+9+0)/3=6.0
# min=4, max=10
_THREE = _ctx((4, 10, 7))

# Uniform: all five sentences identical length
# mean=8.0, variance=0.0, min=8, max=8
_UNIFORM_LOCAL = _ctx((10, 10, 10, 10, 10))

# Variable: wildly different lengths
# mean=45/5=9.0, variance=336/5=67.2, min=2, max=20
_VARIABLE_LOCAL = _ctx((2, 18, 3, 20, 2))

# Turkish minimal context for language-agnosticism test
_TR_CTX = AnalysisContext(
    raw_text="stub",
    language=Language.TURKISH,
    document_type=DocumentType.ACADEMIC,
    cleaned_text="stub",
    tokens=("bir", "iki", "üç", "dört", "beş", "altı", "yedi"),
    sentences=("S0.", "S1."),
    stems=("bir", "iki", "üç", "dört", "beş", "altı", "yedi"),
    sentence_token_counts=(3, 4),
    # mean=3.5, variance=((3-3.5)²+(4-3.5)²)/2=(0.25+0.25)/2=0.25
)


# ---------------------------------------------------------------------------
# Shared analyzer instance (stateless — safe to reuse)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analyzer() -> SentenceStatisticsAnalyzer:
    """Shared SentenceStatisticsAnalyzer; stateless so module scope is safe."""
    return SentenceStatisticsAnalyzer()


# ---------------------------------------------------------------------------
# Analyzer identity
# ---------------------------------------------------------------------------


class TestAnalyzerIdentity:
    def test_name_is_sentence_stats(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.name == "sentence_stats"

    def test_analyze_returns_sentence_stats_instance(
        self, analyzer: SentenceStatisticsAnalyzer
    ):
        assert isinstance(analyzer.analyze(_SINGLE), SentenceStats)


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_total_sentences_is_zero(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.analyze(_EMPTY).total_sentences == 0

    def test_avg_sentence_length_is_zero(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.analyze(_EMPTY).avg_sentence_length == 0.0

    def test_variance_is_zero(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.analyze(_EMPTY).sentence_length_variance == 0.0

    def test_min_is_zero(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.analyze(_EMPTY).min_sentence_length == 0

    def test_max_is_zero(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.analyze(_EMPTY).max_sentence_length == 0

    def test_returns_sentence_stats(self, analyzer: SentenceStatisticsAnalyzer):
        assert isinstance(analyzer.analyze(_EMPTY), SentenceStats)


# ---------------------------------------------------------------------------
# Single sentence
# ---------------------------------------------------------------------------


class TestSingleSentence:
    def test_total_sentences_is_one(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.analyze(_SINGLE).total_sentences == 1

    def test_avg_equals_token_count(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.analyze(_SINGLE).avg_sentence_length == pytest.approx(5.0)

    def test_variance_is_zero(self, analyzer: SentenceStatisticsAnalyzer):
        # Population variance of a single value is always 0
        assert analyzer.analyze(_SINGLE).sentence_length_variance == pytest.approx(0.0)

    def test_min_equals_max(self, analyzer: SentenceStatisticsAnalyzer):
        result = analyzer.analyze(_SINGLE)
        assert result.min_sentence_length == result.max_sentence_length

    def test_min_equals_token_count(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.analyze(_SINGLE).min_sentence_length == 5

    def test_max_equals_token_count(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.analyze(_SINGLE).max_sentence_length == 5


# ---------------------------------------------------------------------------
# Two equal-length sentences
# ---------------------------------------------------------------------------


class TestTwoEqualSentences:
    def test_total_sentences(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.analyze(_TWO_EQUAL).total_sentences == 2

    def test_avg_is_shared_length(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.analyze(_TWO_EQUAL).avg_sentence_length == pytest.approx(8.0)

    def test_variance_is_zero(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.analyze(_TWO_EQUAL).sentence_length_variance == pytest.approx(0.0)

    def test_min_equals_max(self, analyzer: SentenceStatisticsAnalyzer):
        result = analyzer.analyze(_TWO_EQUAL)
        assert result.min_sentence_length == result.max_sentence_length == 8


# ---------------------------------------------------------------------------
# Exact arithmetic — three sentences with different lengths (4, 10, 7)
# ---------------------------------------------------------------------------


class TestThreeSentenceExact:
    def test_total_sentences(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.analyze(_THREE).total_sentences == 3

    def test_avg(self, analyzer: SentenceStatisticsAnalyzer):
        # (4+10+7)/3 = 21/3 = 7.0
        assert analyzer.analyze(_THREE).avg_sentence_length == pytest.approx(7.0)

    def test_variance(self, analyzer: SentenceStatisticsAnalyzer):
        # ((4-7)²+(10-7)²+(7-7)²)/3 = (9+9+0)/3 = 6.0
        assert analyzer.analyze(_THREE).sentence_length_variance == pytest.approx(6.0)

    def test_min(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.analyze(_THREE).min_sentence_length == 4

    def test_max(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.analyze(_THREE).max_sentence_length == 10


# ---------------------------------------------------------------------------
# Exact arithmetic — Turkish two-sentence context (3, 4)
# ---------------------------------------------------------------------------


class TestTurkishContext:
    def test_total_sentences(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.analyze(_TR_CTX).total_sentences == 2

    def test_avg(self, analyzer: SentenceStatisticsAnalyzer):
        # (3+4)/2 = 3.5
        assert analyzer.analyze(_TR_CTX).avg_sentence_length == pytest.approx(3.5)

    def test_variance(self, analyzer: SentenceStatisticsAnalyzer):
        # ((3-3.5)²+(4-3.5)²)/2 = (0.25+0.25)/2 = 0.25
        assert analyzer.analyze(_TR_CTX).sentence_length_variance == pytest.approx(0.25)

    def test_min(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.analyze(_TR_CTX).min_sentence_length == 3

    def test_max(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.analyze(_TR_CTX).max_sentence_length == 4


# ---------------------------------------------------------------------------
# Uniform: all sentences identical length (10, 10, 10, 10, 10)
# ---------------------------------------------------------------------------


class TestUniformLocal:
    def test_variance_is_zero(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.analyze(_UNIFORM_LOCAL).sentence_length_variance == pytest.approx(0.0)

    def test_min_equals_max(self, analyzer: SentenceStatisticsAnalyzer):
        result = analyzer.analyze(_UNIFORM_LOCAL)
        assert result.min_sentence_length == result.max_sentence_length

    def test_avg_equals_uniform_length(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.analyze(_UNIFORM_LOCAL).avg_sentence_length == pytest.approx(10.0)

    def test_total_sentences(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.analyze(_UNIFORM_LOCAL).total_sentences == 5


# ---------------------------------------------------------------------------
# Variable: sentence counts (2, 18, 3, 20, 2)
# mean=9.0, variance=67.2, min=2, max=20
# ---------------------------------------------------------------------------


class TestVariableLocal:
    def test_total_sentences(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.analyze(_VARIABLE_LOCAL).total_sentences == 5

    def test_avg(self, analyzer: SentenceStatisticsAnalyzer):
        # (2+18+3+20+2)/5 = 45/5 = 9.0
        assert analyzer.analyze(_VARIABLE_LOCAL).avg_sentence_length == pytest.approx(9.0)

    def test_variance(self, analyzer: SentenceStatisticsAnalyzer):
        # (49+81+36+121+49)/5 = 336/5 = 67.2
        assert analyzer.analyze(_VARIABLE_LOCAL).sentence_length_variance == pytest.approx(67.2)

    def test_min(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.analyze(_VARIABLE_LOCAL).min_sentence_length == 2

    def test_max(self, analyzer: SentenceStatisticsAnalyzer):
        assert analyzer.analyze(_VARIABLE_LOCAL).max_sentence_length == 20

    def test_variance_exceeds_uniform(self, analyzer: SentenceStatisticsAnalyzer):
        v_var = analyzer.analyze(_VARIABLE_LOCAL).sentence_length_variance
        u_var = analyzer.analyze(_UNIFORM_LOCAL).sentence_length_variance
        assert v_var > u_var


# ---------------------------------------------------------------------------
# Structural invariants (hold for any valid context)
# ---------------------------------------------------------------------------


class TestInvariants:
    @pytest.mark.parametrize("ctx", [
        _EMPTY, _SINGLE, _TWO_EQUAL, _THREE, _UNIFORM_LOCAL, _VARIABLE_LOCAL, _TR_CTX,
    ])
    def test_total_sentences_matches_counts_length(
        self, analyzer: SentenceStatisticsAnalyzer, ctx: AnalysisContext
    ):
        result = analyzer.analyze(ctx)
        assert result.total_sentences == len(ctx.sentence_token_counts)

    @pytest.mark.parametrize("ctx", [
        _EMPTY, _SINGLE, _TWO_EQUAL, _THREE, _UNIFORM_LOCAL, _VARIABLE_LOCAL, _TR_CTX,
    ])
    def test_min_not_exceeds_max(
        self, analyzer: SentenceStatisticsAnalyzer, ctx: AnalysisContext
    ):
        result = analyzer.analyze(ctx)
        assert result.min_sentence_length <= result.max_sentence_length

    @pytest.mark.parametrize("ctx", [
        _EMPTY, _SINGLE, _TWO_EQUAL, _THREE, _UNIFORM_LOCAL, _VARIABLE_LOCAL, _TR_CTX,
    ])
    def test_avg_between_min_and_max(
        self, analyzer: SentenceStatisticsAnalyzer, ctx: AnalysisContext
    ):
        result = analyzer.analyze(ctx)
        assert result.min_sentence_length <= result.avg_sentence_length <= result.max_sentence_length

    @pytest.mark.parametrize("ctx", [
        _EMPTY, _SINGLE, _TWO_EQUAL, _THREE, _UNIFORM_LOCAL, _VARIABLE_LOCAL, _TR_CTX,
    ])
    def test_variance_non_negative(
        self, analyzer: SentenceStatisticsAnalyzer, ctx: AnalysisContext
    ):
        assert analyzer.analyze(ctx).sentence_length_variance >= 0.0

    @pytest.mark.parametrize("ctx", [
        _EMPTY, _SINGLE, _TWO_EQUAL, _THREE, _UNIFORM_LOCAL, _VARIABLE_LOCAL, _TR_CTX,
    ])
    def test_pydantic_model_validates(
        self, analyzer: SentenceStatisticsAnalyzer, ctx: AnalysisContext
    ):
        result = analyzer.analyze(ctx)
        assert SentenceStats.model_validate(result.model_dump())


# ---------------------------------------------------------------------------
# Conftest fixture: uniform_sentence_context
# sentence_token_counts = (8, 8, 8, 8, 8)
# ---------------------------------------------------------------------------


class TestUniformFixture:
    def test_total_sentences(
        self, analyzer: SentenceStatisticsAnalyzer, uniform_sentence_context: AnalysisContext
    ):
        assert analyzer.analyze(uniform_sentence_context).total_sentences == 5

    def test_avg_sentence_length(
        self, analyzer: SentenceStatisticsAnalyzer, uniform_sentence_context: AnalysisContext
    ):
        assert analyzer.analyze(uniform_sentence_context).avg_sentence_length == pytest.approx(8.0)

    def test_variance_is_zero(
        self, analyzer: SentenceStatisticsAnalyzer, uniform_sentence_context: AnalysisContext
    ):
        assert analyzer.analyze(uniform_sentence_context).sentence_length_variance == pytest.approx(0.0)

    def test_min_equals_max(
        self, analyzer: SentenceStatisticsAnalyzer, uniform_sentence_context: AnalysisContext
    ):
        result = analyzer.analyze(uniform_sentence_context)
        assert result.min_sentence_length == result.max_sentence_length == 8

    def test_variance_lower_than_variable(
        self,
        analyzer: SentenceStatisticsAnalyzer,
        uniform_sentence_context: AnalysisContext,
        variable_sentence_context: AnalysisContext,
    ):
        u = analyzer.analyze(uniform_sentence_context).sentence_length_variance
        v = analyzer.analyze(variable_sentence_context).sentence_length_variance
        assert u < v


# ---------------------------------------------------------------------------
# Conftest fixture: variable_sentence_context
# sentence_token_counts = (2, 18, 3, 20, 2), variance = 67.2
# ---------------------------------------------------------------------------


class TestVariableFixture:
    def test_total_sentences(
        self, analyzer: SentenceStatisticsAnalyzer, variable_sentence_context: AnalysisContext
    ):
        assert analyzer.analyze(variable_sentence_context).total_sentences == 5

    def test_avg_sentence_length(
        self, analyzer: SentenceStatisticsAnalyzer, variable_sentence_context: AnalysisContext
    ):
        # (2+18+3+20+2)/5 = 9.0
        assert analyzer.analyze(variable_sentence_context).avg_sentence_length == pytest.approx(9.0)

    def test_variance(
        self, analyzer: SentenceStatisticsAnalyzer, variable_sentence_context: AnalysisContext
    ):
        # 336/5 = 67.2
        assert analyzer.analyze(variable_sentence_context).sentence_length_variance == pytest.approx(67.2)

    def test_min_sentence_length(
        self, analyzer: SentenceStatisticsAnalyzer, variable_sentence_context: AnalysisContext
    ):
        assert analyzer.analyze(variable_sentence_context).min_sentence_length == 2

    def test_max_sentence_length(
        self, analyzer: SentenceStatisticsAnalyzer, variable_sentence_context: AnalysisContext
    ):
        assert analyzer.analyze(variable_sentence_context).max_sentence_length == 20

    def test_large_spread_between_min_and_max(
        self, analyzer: SentenceStatisticsAnalyzer, variable_sentence_context: AnalysisContext
    ):
        result = analyzer.analyze(variable_sentence_context)
        assert (result.max_sentence_length - result.min_sentence_length) >= 15

    def test_variance_substantially_above_zero(
        self, analyzer: SentenceStatisticsAnalyzer, variable_sentence_context: AnalysisContext
    ):
        assert analyzer.analyze(variable_sentence_context).sentence_length_variance > 50.0


# ---------------------------------------------------------------------------
# Realistic fixtures from conftest
# ---------------------------------------------------------------------------


class TestRealisticEnglish:
    def test_returns_sentence_stats(
        self, analyzer: SentenceStatisticsAnalyzer, en_analysis_context: AnalysisContext
    ):
        assert isinstance(analyzer.analyze(en_analysis_context), SentenceStats)

    def test_total_sentences_matches_context(
        self, analyzer: SentenceStatisticsAnalyzer, en_analysis_context: AnalysisContext
    ):
        result = analyzer.analyze(en_analysis_context)
        assert result.total_sentences == len(en_analysis_context.sentences)

    def test_avg_within_plausible_range(
        self, analyzer: SentenceStatisticsAnalyzer, en_analysis_context: AnalysisContext
    ):
        avg = analyzer.analyze(en_analysis_context).avg_sentence_length
        assert 1.0 <= avg <= 50.0

    def test_min_not_exceeds_avg(
        self, analyzer: SentenceStatisticsAnalyzer, en_analysis_context: AnalysisContext
    ):
        result = analyzer.analyze(en_analysis_context)
        assert result.min_sentence_length <= result.avg_sentence_length


class TestRealisticTurkish:
    def test_returns_sentence_stats(
        self, analyzer: SentenceStatisticsAnalyzer, tr_analysis_context: AnalysisContext
    ):
        assert isinstance(analyzer.analyze(tr_analysis_context), SentenceStats)

    def test_total_sentences_matches_context(
        self, analyzer: SentenceStatisticsAnalyzer, tr_analysis_context: AnalysisContext
    ):
        result = analyzer.analyze(tr_analysis_context)
        assert result.total_sentences == len(tr_analysis_context.sentences)

    def test_avg_positive(
        self, analyzer: SentenceStatisticsAnalyzer, tr_analysis_context: AnalysisContext
    ):
        assert analyzer.analyze(tr_analysis_context).avg_sentence_length > 0.0
