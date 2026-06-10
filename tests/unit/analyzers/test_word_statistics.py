"""Unit tests for WordStatisticsAnalyzer."""

import pytest

from src.analyzers.word_statistics import WordStatisticsAnalyzer
from src.models.analysis import AnalysisContext
from src.models.enums import DocumentType, Language
from src.models.response import WordStats

# ---------------------------------------------------------------------------
# Module-level hand-crafted contexts for exact-value assertions.
# These are deliberately minimal so expected values can be computed by
# inspection without running any external code.
# ---------------------------------------------------------------------------

# 4 tokens, all map to distinct stems → diversity = 1.0
# surface lengths: "the"=3, "cats"=4, "run"=3, "fast"=4 → avg = 14/4 = 3.5
_ALL_UNIQUE = AnalysisContext(
    raw_text="The cats run fast.",
    language=Language.ENGLISH,
    document_type=DocumentType.ESSAY,
    cleaned_text="The cats run fast.",
    tokens=("the", "cats", "run", "fast"),
    sentences=("The cats run fast.",),
    stems=("the", "cat", "run", "fast"),
    sentence_token_counts=(4,),
)

# 6 tokens, "the" stem repeated twice → 5 unique stems, diversity = 5/6
# surface lengths: "the"=3,"cat"=3,"sat"=3,"on"=2,"the"=3,"mat"=3 → sum=17, avg=17/6
_ONE_REPEAT = AnalysisContext(
    raw_text="The cat sat on the mat.",
    language=Language.ENGLISH,
    document_type=DocumentType.ESSAY,
    cleaned_text="The cat sat on the mat.",
    tokens=("the", "cat", "sat", "on", "the", "mat"),
    sentences=("The cat sat on the mat.",),
    stems=("the", "cat", "sat", "on", "the", "mat"),
    sentence_token_counts=(6,),
)

# 5 tokens, all reduce to the same stem → 1 unique, diversity = 1/5 = 0.2
# surface lengths: "go"=2,"goes"=4,"went"=4,"going"=5,"gone"=4 → sum=19, avg=19/5=3.8
_ALL_SAME_STEM = AnalysisContext(
    raw_text="go goes went going gone",
    language=Language.ENGLISH,
    document_type=DocumentType.ESSAY,
    cleaned_text="go goes went going gone",
    tokens=("go", "goes", "went", "going", "gone"),
    sentences=("go goes went going gone",),
    stems=("go", "go", "go", "go", "go"),
    sentence_token_counts=(5,),
)

# 1 token — edge-case singleton
# surface length: "hello"=5 → avg = 5.0
_SINGLE_TOKEN = AnalysisContext(
    raw_text="Hello.",
    language=Language.ENGLISH,
    document_type=DocumentType.ESSAY,
    cleaned_text="Hello.",
    tokens=("hello",),
    sentences=("Hello.",),
    stems=("hello",),
    sentence_token_counts=(1,),
)

# 0 tokens — fully empty context
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

# Turkish: 4 tokens, distinct surface lengths demonstrating agglutinative morphology.
# surfaces: "öğrenci"=7, "öğrenciler"=10, "öğrencilerin"=12, "öğrencilik"=10 → sum=39, avg=9.75
# stems: all → "öğrenci" → 1 unique, diversity = 1/4 = 0.25
_TR_SAME_STEM = AnalysisContext(
    raw_text="öğrenci öğrenciler öğrencilerin öğrencilik",
    language=Language.TURKISH,
    document_type=DocumentType.ESSAY,
    cleaned_text="öğrenci öğrenciler öğrencilerin öğrencilik",
    tokens=("öğrenci", "öğrenciler", "öğrencilerin", "öğrencilik"),
    sentences=("öğrenci öğrenciler öğrencilerin öğrencilik",),
    stems=("öğrenci", "öğrenci", "öğrenci", "öğrenci"),
    sentence_token_counts=(4,),
)


# ---------------------------------------------------------------------------
# Shared analyzer instance (stateless — safe to reuse)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def analyzer() -> WordStatisticsAnalyzer:
    """Shared WordStatisticsAnalyzer; stateless so module scope is safe."""
    return WordStatisticsAnalyzer()


# ---------------------------------------------------------------------------
# Analyzer identity
# ---------------------------------------------------------------------------


class TestAnalyzerIdentity:
    def test_name_is_word_stats(self, analyzer: WordStatisticsAnalyzer):
        assert analyzer.name == "word_stats"

    def test_analyze_returns_word_stats_instance(self, analyzer: WordStatisticsAnalyzer):
        result = analyzer.analyze(_ALL_UNIQUE)
        assert isinstance(result, WordStats)


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_total_words_is_zero(self, analyzer: WordStatisticsAnalyzer):
        assert analyzer.analyze(_EMPTY).total_words == 0

    def test_unique_words_is_zero(self, analyzer: WordStatisticsAnalyzer):
        assert analyzer.analyze(_EMPTY).unique_words == 0

    def test_lexical_diversity_is_zero(self, analyzer: WordStatisticsAnalyzer):
        assert analyzer.analyze(_EMPTY).lexical_diversity == 0.0

    def test_avg_word_length_is_zero(self, analyzer: WordStatisticsAnalyzer):
        assert analyzer.analyze(_EMPTY).avg_word_length == 0.0

    def test_returns_word_stats(self, analyzer: WordStatisticsAnalyzer):
        assert isinstance(analyzer.analyze(_EMPTY), WordStats)


# ---------------------------------------------------------------------------
# Single token
# ---------------------------------------------------------------------------


class TestSingleToken:
    def test_total_words_is_one(self, analyzer: WordStatisticsAnalyzer):
        assert analyzer.analyze(_SINGLE_TOKEN).total_words == 1

    def test_unique_words_is_one(self, analyzer: WordStatisticsAnalyzer):
        assert analyzer.analyze(_SINGLE_TOKEN).unique_words == 1

    def test_lexical_diversity_is_one(self, analyzer: WordStatisticsAnalyzer):
        assert analyzer.analyze(_SINGLE_TOKEN).lexical_diversity == pytest.approx(1.0)

    def test_avg_word_length_equals_token_length(self, analyzer: WordStatisticsAnalyzer):
        assert analyzer.analyze(_SINGLE_TOKEN).avg_word_length == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Exact-value arithmetic: all-unique stems
# ---------------------------------------------------------------------------


class TestAllUniqueStems:
    def test_total_words(self, analyzer: WordStatisticsAnalyzer):
        assert analyzer.analyze(_ALL_UNIQUE).total_words == 4

    def test_unique_words(self, analyzer: WordStatisticsAnalyzer):
        assert analyzer.analyze(_ALL_UNIQUE).unique_words == 4

    def test_lexical_diversity_is_one(self, analyzer: WordStatisticsAnalyzer):
        assert analyzer.analyze(_ALL_UNIQUE).lexical_diversity == pytest.approx(1.0)

    def test_avg_word_length(self, analyzer: WordStatisticsAnalyzer):
        # "the"=3, "cats"=4, "run"=3, "fast"=4 → 14/4 = 3.5
        assert analyzer.analyze(_ALL_UNIQUE).avg_word_length == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# Exact-value arithmetic: one repeated stem
# ---------------------------------------------------------------------------


class TestOneRepeatStem:
    def test_total_words(self, analyzer: WordStatisticsAnalyzer):
        assert analyzer.analyze(_ONE_REPEAT).total_words == 6

    def test_unique_words(self, analyzer: WordStatisticsAnalyzer):
        # "the" appears twice but "the" stem de-duplcates to 1 → 5 unique
        assert analyzer.analyze(_ONE_REPEAT).unique_words == 5

    def test_lexical_diversity(self, analyzer: WordStatisticsAnalyzer):
        # 5/6
        assert analyzer.analyze(_ONE_REPEAT).lexical_diversity == pytest.approx(5 / 6)

    def test_avg_word_length(self, analyzer: WordStatisticsAnalyzer):
        # "the"=3,"cat"=3,"sat"=3,"on"=2,"the"=3,"mat"=3 → 17/6
        assert analyzer.analyze(_ONE_REPEAT).avg_word_length == pytest.approx(17 / 6)


# ---------------------------------------------------------------------------
# Exact-value arithmetic: all tokens collapse to the same stem
# ---------------------------------------------------------------------------


class TestAllSameStem:
    def test_total_words(self, analyzer: WordStatisticsAnalyzer):
        assert analyzer.analyze(_ALL_SAME_STEM).total_words == 5

    def test_unique_words_is_one(self, analyzer: WordStatisticsAnalyzer):
        assert analyzer.analyze(_ALL_SAME_STEM).unique_words == 1

    def test_lexical_diversity_is_one_fifth(self, analyzer: WordStatisticsAnalyzer):
        assert analyzer.analyze(_ALL_SAME_STEM).lexical_diversity == pytest.approx(0.2)

    def test_avg_word_length(self, analyzer: WordStatisticsAnalyzer):
        # "go"=2,"goes"=4,"went"=4,"going"=5,"gone"=4 → 19/5 = 3.8
        assert analyzer.analyze(_ALL_SAME_STEM).avg_word_length == pytest.approx(3.8)


# ---------------------------------------------------------------------------
# Turkish — same stem, agglutinative surface forms
# ---------------------------------------------------------------------------


class TestTurkishSameStem:
    def test_total_words(self, analyzer: WordStatisticsAnalyzer):
        assert analyzer.analyze(_TR_SAME_STEM).total_words == 4

    def test_unique_words_is_one(self, analyzer: WordStatisticsAnalyzer):
        assert analyzer.analyze(_TR_SAME_STEM).unique_words == 1

    def test_lexical_diversity_is_one_quarter(self, analyzer: WordStatisticsAnalyzer):
        assert analyzer.analyze(_TR_SAME_STEM).lexical_diversity == pytest.approx(0.25)

    def test_avg_word_length(self, analyzer: WordStatisticsAnalyzer):
        # 7+10+12+10 = 39, avg = 39/4 = 9.75
        assert analyzer.analyze(_TR_SAME_STEM).avg_word_length == pytest.approx(9.75)

    def test_turkish_avg_length_reflects_agglutination(self, analyzer: WordStatisticsAnalyzer):
        tr_avg = analyzer.analyze(_TR_SAME_STEM).avg_word_length
        en_avg = analyzer.analyze(_ALL_UNIQUE).avg_word_length
        assert tr_avg > en_avg


# ---------------------------------------------------------------------------
# Structural invariants (hold for any valid context)
# ---------------------------------------------------------------------------


class TestInvariants:
    @pytest.mark.parametrize("ctx", [
        _EMPTY, _SINGLE_TOKEN, _ALL_UNIQUE, _ONE_REPEAT, _ALL_SAME_STEM, _TR_SAME_STEM,
    ])
    def test_unique_never_exceeds_total(
        self, analyzer: WordStatisticsAnalyzer, ctx: AnalysisContext
    ):
        result = analyzer.analyze(ctx)
        assert result.unique_words <= result.total_words

    @pytest.mark.parametrize("ctx", [
        _EMPTY, _SINGLE_TOKEN, _ALL_UNIQUE, _ONE_REPEAT, _ALL_SAME_STEM, _TR_SAME_STEM,
    ])
    def test_diversity_in_unit_interval(
        self, analyzer: WordStatisticsAnalyzer, ctx: AnalysisContext
    ):
        result = analyzer.analyze(ctx)
        assert 0.0 <= result.lexical_diversity <= 1.0

    @pytest.mark.parametrize("ctx", [
        _EMPTY, _SINGLE_TOKEN, _ALL_UNIQUE, _ONE_REPEAT, _ALL_SAME_STEM, _TR_SAME_STEM,
    ])
    def test_avg_word_length_non_negative(
        self, analyzer: WordStatisticsAnalyzer, ctx: AnalysisContext
    ):
        result = analyzer.analyze(ctx)
        assert result.avg_word_length >= 0.0

    @pytest.mark.parametrize("ctx", [
        _EMPTY, _SINGLE_TOKEN, _ALL_UNIQUE, _ONE_REPEAT, _ALL_SAME_STEM, _TR_SAME_STEM,
    ])
    def test_pydantic_model_validates_without_error(
        self, analyzer: WordStatisticsAnalyzer, ctx: AnalysisContext
    ):
        result = analyzer.analyze(ctx)
        assert WordStats.model_validate(result.model_dump())


# ---------------------------------------------------------------------------
# Realistic English fixture (en_analysis_context from conftest)
# ---------------------------------------------------------------------------


class TestRealisticEnglish:
    def test_returns_word_stats(
        self, analyzer: WordStatisticsAnalyzer, en_analysis_context: AnalysisContext
    ):
        assert isinstance(analyzer.analyze(en_analysis_context), WordStats)

    def test_total_words_matches_token_count(
        self, analyzer: WordStatisticsAnalyzer, en_analysis_context: AnalysisContext
    ):
        result = analyzer.analyze(en_analysis_context)
        assert result.total_words == len(en_analysis_context.tokens)

    def test_unique_words_not_exceeds_total(
        self, analyzer: WordStatisticsAnalyzer, en_analysis_context: AnalysisContext
    ):
        result = analyzer.analyze(en_analysis_context)
        assert result.unique_words <= result.total_words

    def test_diversity_in_valid_range(
        self, analyzer: WordStatisticsAnalyzer, en_analysis_context: AnalysisContext
    ):
        result = analyzer.analyze(en_analysis_context)
        assert 0.0 <= result.lexical_diversity <= 1.0

    def test_avg_length_positive(
        self, analyzer: WordStatisticsAnalyzer, en_analysis_context: AnalysisContext
    ):
        assert analyzer.analyze(en_analysis_context).avg_word_length > 0.0

    def test_unique_uses_stems_not_surface_forms(
        self, analyzer: WordStatisticsAnalyzer, en_analysis_context: AnalysisContext
    ):
        result = analyzer.analyze(en_analysis_context)
        surface_unique = len(set(en_analysis_context.tokens))
        stem_unique = len(set(en_analysis_context.stems))
        assert result.unique_words == stem_unique
        # Stems collapse inflections → stem unique ≤ surface unique
        assert stem_unique <= surface_unique


# ---------------------------------------------------------------------------
# Realistic Turkish fixture (tr_analysis_context from conftest)
# ---------------------------------------------------------------------------


class TestRealisticTurkish:
    def test_returns_word_stats(
        self, analyzer: WordStatisticsAnalyzer, tr_analysis_context: AnalysisContext
    ):
        assert isinstance(analyzer.analyze(tr_analysis_context), WordStats)

    def test_total_words_matches_token_count(
        self, analyzer: WordStatisticsAnalyzer, tr_analysis_context: AnalysisContext
    ):
        result = analyzer.analyze(tr_analysis_context)
        assert result.total_words == len(tr_analysis_context.tokens)

    def test_unique_words_not_exceeds_total(
        self, analyzer: WordStatisticsAnalyzer, tr_analysis_context: AnalysisContext
    ):
        result = analyzer.analyze(tr_analysis_context)
        assert result.unique_words <= result.total_words

    def test_diversity_in_valid_range(
        self, analyzer: WordStatisticsAnalyzer, tr_analysis_context: AnalysisContext
    ):
        result = analyzer.analyze(tr_analysis_context)
        assert 0.0 <= result.lexical_diversity <= 1.0

    def test_avg_length_positive(
        self, analyzer: WordStatisticsAnalyzer, tr_analysis_context: AnalysisContext
    ):
        assert analyzer.analyze(tr_analysis_context).avg_word_length > 0.0

    def test_stems_reduce_unique_count(
        self, analyzer: WordStatisticsAnalyzer, tr_analysis_context: AnalysisContext
    ):
        result = analyzer.analyze(tr_analysis_context)
        surface_unique = len(set(tr_analysis_context.tokens))
        assert result.unique_words <= surface_unique


# ---------------------------------------------------------------------------
# Rich English fixture (en_rich_context from conftest) — known exact values
# ---------------------------------------------------------------------------


class TestRichEnglish:
    def test_total_words(
        self, analyzer: WordStatisticsAnalyzer, en_rich_context: AnalysisContext
    ):
        # 23 tokens defined in the fixture
        assert analyzer.analyze(en_rich_context).total_words == 23

    def test_unique_words(
        self, analyzer: WordStatisticsAnalyzer, en_rich_context: AnalysisContext
    ):
        # 19 unique stems: digit(×3), educ(×2), student(×2) repeated
        assert analyzer.analyze(en_rich_context).unique_words == 19

    def test_lexical_diversity(
        self, analyzer: WordStatisticsAnalyzer, en_rich_context: AnalysisContext
    ):
        assert analyzer.analyze(en_rich_context).lexical_diversity == pytest.approx(19 / 23)

    def test_avg_word_length(
        self, analyzer: WordStatisticsAnalyzer, en_rich_context: AnalysisContext
    ):
        # Surface lengths sum to 168 across 23 tokens
        assert analyzer.analyze(en_rich_context).avg_word_length == pytest.approx(168 / 23)

    def test_avg_length_plausible_for_academic_english(
        self, analyzer: WordStatisticsAnalyzer, en_rich_context: AnalysisContext
    ):
        avg = analyzer.analyze(en_rich_context).avg_word_length
        assert 4.0 <= avg <= 12.0


# ---------------------------------------------------------------------------
# Rich Turkish fixture (tr_rich_context from conftest) — known exact values
# ---------------------------------------------------------------------------


class TestRichTurkish:
    def test_total_words(
        self, analyzer: WordStatisticsAnalyzer, tr_rich_context: AnalysisContext
    ):
        # 20 tokens defined in the fixture
        assert analyzer.analyze(tr_rich_context).total_words == 20

    def test_unique_words(
        self, analyzer: WordStatisticsAnalyzer, tr_rich_context: AnalysisContext
    ):
        # 17 unique stems: dijital(×2), teknoloji(×2), eğitim(×2) repeated
        assert analyzer.analyze(tr_rich_context).unique_words == 17

    def test_lexical_diversity(
        self, analyzer: WordStatisticsAnalyzer, tr_rich_context: AnalysisContext
    ):
        assert analyzer.analyze(tr_rich_context).lexical_diversity == pytest.approx(0.85)

    def test_avg_word_length(
        self, analyzer: WordStatisticsAnalyzer, tr_rich_context: AnalysisContext
    ):
        # Surface lengths sum to 184 across 20 tokens
        assert analyzer.analyze(tr_rich_context).avg_word_length == pytest.approx(9.2)

    def test_avg_length_plausible_for_agglutinative_turkish(
        self, analyzer: WordStatisticsAnalyzer, tr_rich_context: AnalysisContext
    ):
        avg = analyzer.analyze(tr_rich_context).avg_word_length
        assert 6.0 <= avg <= 15.0

    def test_turkish_words_longer_than_english_on_average(
        self,
        analyzer: WordStatisticsAnalyzer,
        en_rich_context: AnalysisContext,
        tr_rich_context: AnalysisContext,
    ):
        en_avg = analyzer.analyze(en_rich_context).avg_word_length
        tr_avg = analyzer.analyze(tr_rich_context).avg_word_length
        assert tr_avg > en_avg
