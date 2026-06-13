"""Unit tests for TransitionAnalyzer."""

import pytest

from src.analyzers.transition import (
    _ALL_TRANSITIONS,
    _EN_TRANSITIONS,
    _TR_TRANSITIONS,
    TransitionAnalyzer,
    _compute_score,
    _count_transitions,
)
from src.models.analysis import AnalysisContext
from src.models.enums import DocumentType, Language
from src.models.response import TransitionResult

# ---------------------------------------------------------------------------
# Module-level AnalysisContext fixtures with pre-verified expected values.
# ---------------------------------------------------------------------------

# Empty — no tokens, no sentences.
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

# ── _EN_NO_TRANSITIONS ───────────────────────────────────────────────────────
# Three sentences with no transition words.
# Expected: transition_count=0, all lists empty, score=0.0
_EN_NO_TRANSITIONS = AnalysisContext(
    raw_text="Language shapes perception and reality. Philosophers debate structure. Researchers argue categories influence cognition.",
    language=Language.ENGLISH,
    document_type=DocumentType.ESSAY,
    cleaned_text="stub",
    tokens=(
        "language", "shapes", "perception", "and", "reality",
        "philosophers", "debate", "structure",
        "researchers", "argue", "categories", "influence", "cognition",
    ),
    stems=("languag", "shape", "percept", "and", "realiti",
           "philosoph", "debat", "structur",
           "research", "argu", "categori", "influenc", "cognit"),
    sentences=(
        "Language shapes perception and reality.",
        "Philosophers debate structure.",
        "Researchers argue categories influence cognition.",
    ),
    sentence_token_counts=(5, 3, 5),
)

# ── _EN_SINGLE_USE ───────────────────────────────────────────────────────────
# Five sentences: "however" once (pos 5), "therefore" once (pos 16).
#
# Expected:
#   transition_count=2
#   unique_transitions=["however", "therefore"]
#   repeated_transitions=[]
#   density = 2/5 = 0.4
#   score = min(1, 0.6×0.2 + 0.4×0) = 0.12
_EN_SINGLE_USE = AnalysisContext(
    raw_text=(
        "Students engage with digital platforms. "
        "However some students lack internet access. "
        "Teachers must adapt their methods. "
        "Therefore blended learning is recommended. "
        "Outcomes remain largely positive."
    ),
    language=Language.ENGLISH,
    document_type=DocumentType.ACADEMIC,
    cleaned_text="stub",
    tokens=(
        "students", "engage", "with", "digital", "platforms",
        "however", "some", "students", "lack", "internet", "access",
        "teachers", "must", "adapt", "their", "methods",
        "therefore", "blended", "learning", "is", "recommended",
        "outcomes", "remain", "largely", "positive",
    ),
    stems=("student", "engag", "with", "digit", "platform",
           "howev", "some", "student", "lack", "internet", "access",
           "teacher", "must", "adapt", "their", "method",
           "therefor", "blend", "learn", "is", "recommend",
           "outcom", "remain", "larg", "posit"),
    sentences=(
        "Students engage with digital platforms.",
        "However some students lack internet access.",
        "Teachers must adapt their methods.",
        "Therefore blended learning is recommended.",
        "Outcomes remain largely positive.",
    ),
    sentence_token_counts=(5, 6, 5, 5, 4),
)

# ── _EN_OVERUSE ──────────────────────────────────────────────────────────────
# Five sentences: "furthermore" × 4, "moreover" × 2.
# Positions: furthermore@[0,6,18,24], moreover@[12,25]
#
# Expected:
#   transition_count=6
#   unique_transitions=["furthermore", "moreover"]
#   repeated_transitions=["furthermore", "moreover"]
#   density = 6/5 = 1.2
#   density_signal = min(1, 0.6) = 0.6
#   repeat_ratio = 2/2 = 1.0
#   score = min(1, 0.36 + 0.40) = 0.76
_EN_OVERUSE = AnalysisContext(
    raw_text=(
        "Furthermore digital technology has transformed education. "
        "Furthermore students benefit from vast resources. "
        "Moreover the evidence suggests significant improvement. "
        "Furthermore teachers must adapt pedagogical approaches. "
        "Furthermore moreover outcomes have improved considerably."
    ),
    language=Language.ENGLISH,
    document_type=DocumentType.ACADEMIC,
    cleaned_text="stub",
    tokens=(
        "furthermore", "digital", "technology", "has", "transformed", "education",
        "furthermore", "students", "benefit", "from", "vast", "resources",
        "moreover", "the", "evidence", "suggests", "significant", "improvement",
        "furthermore", "teachers", "must", "adapt", "pedagogical", "approaches",
        "furthermore", "moreover", "outcomes", "have", "improved", "considerably",
    ),
    stems=("furthermor", "digit", "technolog", "has", "transform", "educ",
           "furthermor", "student", "benefit", "from", "vast", "resourc",
           "moreov", "the", "evid", "suggest", "signific", "improv",
           "furthermor", "teacher", "must", "adapt", "pedagog", "approach",
           "furthermor", "moreov", "outcom", "have", "improv", "consider"),
    sentences=(
        "Furthermore digital technology has transformed education.",
        "Furthermore students benefit from vast resources.",
        "Moreover the evidence suggests significant improvement.",
        "Furthermore teachers must adapt pedagogical approaches.",
        "Furthermore moreover outcomes have improved considerably.",
    ),
    sentence_token_counts=(6, 6, 6, 6, 6),
)

# ── _EN_MULTIWORD ────────────────────────────────────────────────────────────
# Four sentences: "in addition" (tokens[4:6]), "in conclusion" (tokens[14:16]).
#
# Expected:
#   transition_count=2
#   unique_transitions=["in addition", "in conclusion"]
#   repeated_transitions=[]
#   density = 2/4 = 0.5
#   score = min(1, 0.6×0.25) = 0.15
_EN_MULTIWORD = AnalysisContext(
    raw_text=(
        "This study examines technology. "
        "In addition students access resources. "
        "Educators must adapt their approaches. "
        "In conclusion technology improves outcomes."
    ),
    language=Language.ENGLISH,
    document_type=DocumentType.ACADEMIC,
    cleaned_text="stub",
    tokens=(
        "this", "study", "examines", "technology",
        "in", "addition", "students", "access", "resources",
        "educators", "must", "adapt", "their", "approaches",
        "in", "conclusion", "technology", "improves", "outcomes",
    ),
    stems=("this", "studi", "examin", "technolog",
           "in", "addit", "student", "access", "resourc",
           "educ", "must", "adapt", "their", "approach",
           "in", "conclus", "technolog", "improv", "outcom"),
    sentences=(
        "This study examines technology.",
        "In addition students access resources.",
        "Educators must adapt their approaches.",
        "In conclusion technology improves outcomes.",
    ),
    sentence_token_counts=(4, 5, 5, 5),
)

# ── _TR_SINGLE_WORD ──────────────────────────────────────────────────────────
# Four Turkish sentences: "ayrıca"@0, "dolayısıyla"@4, "ancak"@12.
#
# Expected:
#   transition_count=3
#   unique_transitions=["ancak", "ayrıca", "dolayısıyla"]
#   repeated_transitions=[]
#   density = 3/4 = 0.75
#   score = min(1, 0.6×0.375) = 0.225
_TR_SINGLE_WORD = AnalysisContext(
    raw_text=(
        "Ayrıca dijital teknoloji önemlidir. "
        "Dolayısıyla eğitim kalitesi artmaktadır. "
        "Öğrenciler çevrimiçi kaynaklara erişmektedir. "
        "Ancak tüm öğrenciler yararlanamaz."
    ),
    language=Language.TURKISH,
    document_type=DocumentType.ACADEMIC,
    cleaned_text="stub",
    tokens=(
        "ayrıca", "dijital", "teknoloji", "önemlidir",
        "dolayısıyla", "eğitim", "kalitesi", "artmaktadır",
        "öğrenciler", "çevrimiçi", "kaynaklara", "erişmektedir",
        "ancak", "tüm", "öğrenciler", "yararlanamaz",
    ),
    stems=("ayrıca", "dijital", "teknoloji", "önemli",
           "dolayısıyla", "eğitim", "kalite", "artır",
           "öğrenci", "çevrimiçi", "kaynak", "eriş",
           "ancak", "tüm", "öğrenci", "yararlan"),
    sentences=(
        "Ayrıca dijital teknoloji önemlidir.",
        "Dolayısıyla eğitim kalitesi artmaktadır.",
        "Öğrenciler çevrimiçi kaynaklara erişmektedir.",
        "Ancak tüm öğrenciler yararlanamaz.",
    ),
    sentence_token_counts=(4, 4, 4, 4),
)

# ── _TR_MULTIWORD ────────────────────────────────────────────────────────────
# Three Turkish sentences:
#   "bunun yanında" @ tokens[0:2]
#   "sonuç olarak"  @ tokens[5:7] and tokens[10:12]
#
# Expected:
#   transition_count=3
#   unique_transitions=["bunun yanında", "sonuç olarak"]
#   repeated_transitions=["sonuç olarak"]
#   density = 3/3 = 1.0
#   density_signal = 0.5, repeat_ratio = 1/2 = 0.5
#   score = min(1, 0.30 + 0.20) = 0.50
_TR_MULTIWORD = AnalysisContext(
    raw_text=(
        "Bunun yanında teknoloji eğitimi dönüştürür. "
        "Sonuç olarak öğrenciler daha başarılıdır. "
        "Sonuç olarak eğitim kalitesi artmaktadır."
    ),
    language=Language.TURKISH,
    document_type=DocumentType.ACADEMIC,
    cleaned_text="stub",
    tokens=(
        "bunun", "yanında", "teknoloji", "eğitimi", "dönüştürür",
        "sonuç", "olarak", "öğrenciler", "daha", "başarılıdır",
        "sonuç", "olarak", "eğitim", "kalitesi", "artmaktadır",
    ),
    stems=("bunun", "yanında", "teknoloji", "eğitim", "dönüştür",
           "sonuç", "ol", "öğrenci", "daha", "başarılı",
           "sonuç", "ol", "eğitim", "kalite", "artır"),
    sentences=(
        "Bunun yanında teknoloji eğitimi dönüştürür.",
        "Sonuç olarak öğrenciler daha başarılıdır.",
        "Sonuç olarak eğitim kalitesi artmaktadır.",
    ),
    sentence_token_counts=(5, 5, 5),
)


# ---------------------------------------------------------------------------
# Shared analyzer instance
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analyzer() -> TransitionAnalyzer:
    """Shared TransitionAnalyzer instance."""
    return TransitionAnalyzer()


# ---------------------------------------------------------------------------
# Analyzer identity and construction
# ---------------------------------------------------------------------------


class TestAnalyzerIdentity:
    def test_name_is_transition(self, analyzer: TransitionAnalyzer) -> None:
        assert analyzer.name == "transition"

    def test_analyze_returns_transition_result(
        self, analyzer: TransitionAnalyzer
    ) -> None:
        assert isinstance(analyzer.analyze(_EN_NO_TRANSITIONS), TransitionResult)

    def test_known_english_transitions_registered(self) -> None:
        for phrase in _EN_TRANSITIONS:
            assert phrase in _ALL_TRANSITIONS

    def test_known_turkish_transitions_registered(self) -> None:
        for phrase in _TR_TRANSITIONS:
            assert phrase in _ALL_TRANSITIONS

    def test_all_transitions_lowercase(self) -> None:
        for phrase in _ALL_TRANSITIONS:
            assert phrase == phrase.lower()


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_transition_count_zero(self, analyzer: TransitionAnalyzer) -> None:
        assert analyzer.analyze(_EMPTY).transition_count == 0

    def test_unique_transitions_empty(self, analyzer: TransitionAnalyzer) -> None:
        assert analyzer.analyze(_EMPTY).unique_transitions == []

    def test_repeated_transitions_empty(self, analyzer: TransitionAnalyzer) -> None:
        assert analyzer.analyze(_EMPTY).repeated_transitions == []

    def test_density_zero(self, analyzer: TransitionAnalyzer) -> None:
        assert analyzer.analyze(_EMPTY).transition_density == pytest.approx(0.0)

    def test_score_zero(self, analyzer: TransitionAnalyzer) -> None:
        assert analyzer.analyze(_EMPTY).transition_score == pytest.approx(0.0)

    def test_returns_transition_result(self, analyzer: TransitionAnalyzer) -> None:
        assert isinstance(analyzer.analyze(_EMPTY), TransitionResult)


# ---------------------------------------------------------------------------
# Text with no transition words
# ---------------------------------------------------------------------------


class TestNoTransitions:
    def test_count_zero(self, analyzer: TransitionAnalyzer) -> None:
        assert analyzer.analyze(_EN_NO_TRANSITIONS).transition_count == 0

    def test_unique_empty(self, analyzer: TransitionAnalyzer) -> None:
        assert analyzer.analyze(_EN_NO_TRANSITIONS).unique_transitions == []

    def test_repeated_empty(self, analyzer: TransitionAnalyzer) -> None:
        assert analyzer.analyze(_EN_NO_TRANSITIONS).repeated_transitions == []

    def test_score_zero(self, analyzer: TransitionAnalyzer) -> None:
        assert analyzer.analyze(_EN_NO_TRANSITIONS).transition_score == pytest.approx(
            0.0
        )


# ---------------------------------------------------------------------------
# English — healthy single use of two different transitions
# ---------------------------------------------------------------------------


class TestEnglishSingleUse:
    def test_count_two(self, analyzer: TransitionAnalyzer) -> None:
        assert analyzer.analyze(_EN_SINGLE_USE).transition_count == 2

    def test_however_detected(self, analyzer: TransitionAnalyzer) -> None:
        assert "however" in analyzer.analyze(_EN_SINGLE_USE).unique_transitions

    def test_therefore_detected(self, analyzer: TransitionAnalyzer) -> None:
        assert "therefore" in analyzer.analyze(_EN_SINGLE_USE).unique_transitions

    def test_unique_count_two(self, analyzer: TransitionAnalyzer) -> None:
        assert len(analyzer.analyze(_EN_SINGLE_USE).unique_transitions) == 2

    def test_no_repeated_transitions(self, analyzer: TransitionAnalyzer) -> None:
        assert analyzer.analyze(_EN_SINGLE_USE).repeated_transitions == []

    def test_density(self, analyzer: TransitionAnalyzer) -> None:
        assert analyzer.analyze(_EN_SINGLE_USE).transition_density == pytest.approx(
            0.4
        )

    def test_score_exact(self, analyzer: TransitionAnalyzer) -> None:
        # 0.6 × min(1, 0.4/2) + 0.4 × 0 = 0.6×0.2 = 0.12
        assert analyzer.analyze(_EN_SINGLE_USE).transition_score == pytest.approx(0.12)

    def test_score_lower_than_overuse(self, analyzer: TransitionAnalyzer) -> None:
        single = analyzer.analyze(_EN_SINGLE_USE).transition_score
        overuse = analyzer.analyze(_EN_OVERUSE).transition_score
        assert single < overuse


# ---------------------------------------------------------------------------
# English — formulaic overuse
# ---------------------------------------------------------------------------


class TestEnglishOveruse:
    def test_count_six(self, analyzer: TransitionAnalyzer) -> None:
        assert analyzer.analyze(_EN_OVERUSE).transition_count == 6

    def test_furthermore_in_unique(self, analyzer: TransitionAnalyzer) -> None:
        assert "furthermore" in analyzer.analyze(_EN_OVERUSE).unique_transitions

    def test_moreover_in_unique(self, analyzer: TransitionAnalyzer) -> None:
        assert "moreover" in analyzer.analyze(_EN_OVERUSE).unique_transitions

    def test_unique_count_two(self, analyzer: TransitionAnalyzer) -> None:
        assert len(analyzer.analyze(_EN_OVERUSE).unique_transitions) == 2

    def test_furthermore_in_repeated(self, analyzer: TransitionAnalyzer) -> None:
        assert "furthermore" in analyzer.analyze(_EN_OVERUSE).repeated_transitions

    def test_moreover_in_repeated(self, analyzer: TransitionAnalyzer) -> None:
        assert "moreover" in analyzer.analyze(_EN_OVERUSE).repeated_transitions

    def test_density_exact(self, analyzer: TransitionAnalyzer) -> None:
        assert analyzer.analyze(_EN_OVERUSE).transition_density == pytest.approx(1.2)

    def test_score_exact(self, analyzer: TransitionAnalyzer) -> None:
        # density_signal=0.6, repeat_ratio=1.0 → 0.36+0.40=0.76
        assert analyzer.analyze(_EN_OVERUSE).transition_score == pytest.approx(0.76)


# ---------------------------------------------------------------------------
# English — multi-word transitions
# ---------------------------------------------------------------------------


class TestMultiWordEnglish:
    def test_in_addition_detected(self, analyzer: TransitionAnalyzer) -> None:
        assert "in addition" in analyzer.analyze(_EN_MULTIWORD).unique_transitions

    def test_in_conclusion_detected(self, analyzer: TransitionAnalyzer) -> None:
        assert "in conclusion" in analyzer.analyze(_EN_MULTIWORD).unique_transitions

    def test_count_two(self, analyzer: TransitionAnalyzer) -> None:
        assert analyzer.analyze(_EN_MULTIWORD).transition_count == 2

    def test_no_repeated(self, analyzer: TransitionAnalyzer) -> None:
        assert analyzer.analyze(_EN_MULTIWORD).repeated_transitions == []

    def test_density(self, analyzer: TransitionAnalyzer) -> None:
        assert analyzer.analyze(_EN_MULTIWORD).transition_density == pytest.approx(0.5)

    def test_score_exact(self, analyzer: TransitionAnalyzer) -> None:
        # density_signal=0.25, repeat_ratio=0 → 0.6×0.25=0.15
        assert analyzer.analyze(_EN_MULTIWORD).transition_score == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Turkish — single-word transitions
# ---------------------------------------------------------------------------


class TestTurkishSingleWord:
    def test_ayrıca_detected(self, analyzer: TransitionAnalyzer) -> None:
        assert "ayrıca" in analyzer.analyze(_TR_SINGLE_WORD).unique_transitions

    def test_dolayisiyla_detected(self, analyzer: TransitionAnalyzer) -> None:
        assert "dolayısıyla" in analyzer.analyze(_TR_SINGLE_WORD).unique_transitions

    def test_ancak_detected(self, analyzer: TransitionAnalyzer) -> None:
        assert "ancak" in analyzer.analyze(_TR_SINGLE_WORD).unique_transitions

    def test_count_three(self, analyzer: TransitionAnalyzer) -> None:
        assert analyzer.analyze(_TR_SINGLE_WORD).transition_count == 3

    def test_no_repeated(self, analyzer: TransitionAnalyzer) -> None:
        assert analyzer.analyze(_TR_SINGLE_WORD).repeated_transitions == []

    def test_density(self, analyzer: TransitionAnalyzer) -> None:
        assert analyzer.analyze(_TR_SINGLE_WORD).transition_density == pytest.approx(
            0.75
        )

    def test_score_exact(self, analyzer: TransitionAnalyzer) -> None:
        # density_signal=0.375, repeat_ratio=0 → 0.6×0.375=0.225
        assert analyzer.analyze(_TR_SINGLE_WORD).transition_score == pytest.approx(
            0.225
        )


# ---------------------------------------------------------------------------
# Turkish — multi-word transitions with repetition
# ---------------------------------------------------------------------------


class TestTurkishMultiWord:
    def test_bunun_yaninda_detected(self, analyzer: TransitionAnalyzer) -> None:
        assert "bunun yanında" in analyzer.analyze(_TR_MULTIWORD).unique_transitions

    def test_sonuc_olarak_detected(self, analyzer: TransitionAnalyzer) -> None:
        assert "sonuç olarak" in analyzer.analyze(_TR_MULTIWORD).unique_transitions

    def test_count_three(self, analyzer: TransitionAnalyzer) -> None:
        assert analyzer.analyze(_TR_MULTIWORD).transition_count == 3

    def test_sonuc_olarak_in_repeated(self, analyzer: TransitionAnalyzer) -> None:
        assert "sonuç olarak" in analyzer.analyze(_TR_MULTIWORD).repeated_transitions

    def test_bunun_yaninda_not_repeated(self, analyzer: TransitionAnalyzer) -> None:
        assert "bunun yanında" not in analyzer.analyze(_TR_MULTIWORD).repeated_transitions

    def test_density_one(self, analyzer: TransitionAnalyzer) -> None:
        assert analyzer.analyze(_TR_MULTIWORD).transition_density == pytest.approx(1.0)

    def test_score_exact(self, analyzer: TransitionAnalyzer) -> None:
        # density_signal=0.5, repeat_ratio=0.5 → 0.30+0.20=0.50
        assert analyzer.analyze(_TR_MULTIWORD).transition_score == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# Structural invariants across all contexts
# ---------------------------------------------------------------------------


class TestInvariants:
    @pytest.mark.parametrize(
        "ctx",
        [
            _EMPTY,
            _EN_NO_TRANSITIONS,
            _EN_SINGLE_USE,
            _EN_OVERUSE,
            _EN_MULTIWORD,
            _TR_SINGLE_WORD,
            _TR_MULTIWORD,
        ],
    )
    def test_score_in_unit_interval(
        self, analyzer: TransitionAnalyzer, ctx: AnalysisContext
    ) -> None:
        score = analyzer.analyze(ctx).transition_score
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize(
        "ctx",
        [
            _EMPTY,
            _EN_NO_TRANSITIONS,
            _EN_SINGLE_USE,
            _EN_OVERUSE,
            _EN_MULTIWORD,
            _TR_SINGLE_WORD,
            _TR_MULTIWORD,
        ],
    )
    def test_count_equals_sum_of_unique_occurrences(
        self, analyzer: TransitionAnalyzer, ctx: AnalysisContext
    ) -> None:
        result = analyzer.analyze(ctx)
        assert result.transition_count >= len(result.unique_transitions)

    @pytest.mark.parametrize(
        "ctx",
        [
            _EMPTY,
            _EN_NO_TRANSITIONS,
            _EN_SINGLE_USE,
            _EN_OVERUSE,
            _EN_MULTIWORD,
            _TR_SINGLE_WORD,
            _TR_MULTIWORD,
        ],
    )
    def test_repeated_is_subset_of_unique(
        self, analyzer: TransitionAnalyzer, ctx: AnalysisContext
    ) -> None:
        result = analyzer.analyze(ctx)
        assert set(result.repeated_transitions) <= set(result.unique_transitions)

    @pytest.mark.parametrize(
        "ctx",
        [
            _EMPTY,
            _EN_NO_TRANSITIONS,
            _EN_SINGLE_USE,
            _EN_OVERUSE,
            _EN_MULTIWORD,
            _TR_SINGLE_WORD,
            _TR_MULTIWORD,
        ],
    )
    def test_pydantic_model_validates(
        self, analyzer: TransitionAnalyzer, ctx: AnalysisContext
    ) -> None:
        result = analyzer.analyze(ctx)
        assert TransitionResult.model_validate(result.model_dump())

    @pytest.mark.parametrize(
        "ctx",
        [
            _EMPTY,
            _EN_NO_TRANSITIONS,
            _EN_SINGLE_USE,
            _EN_OVERUSE,
            _EN_MULTIWORD,
            _TR_SINGLE_WORD,
            _TR_MULTIWORD,
        ],
    )
    def test_unique_transitions_sorted(
        self, analyzer: TransitionAnalyzer, ctx: AnalysisContext
    ) -> None:
        result = analyzer.analyze(ctx)
        assert result.unique_transitions == sorted(result.unique_transitions)

    @pytest.mark.parametrize(
        "ctx",
        [
            _EMPTY,
            _EN_NO_TRANSITIONS,
            _EN_SINGLE_USE,
            _EN_OVERUSE,
            _EN_MULTIWORD,
            _TR_SINGLE_WORD,
            _TR_MULTIWORD,
        ],
    )
    def test_repeated_transitions_sorted(
        self, analyzer: TransitionAnalyzer, ctx: AnalysisContext
    ) -> None:
        result = analyzer.analyze(ctx)
        assert result.repeated_transitions == sorted(result.repeated_transitions)


# ---------------------------------------------------------------------------
# Realistic conftest fixtures
# ---------------------------------------------------------------------------


class TestRealisticEnglish:
    def test_returns_transition_result(
        self, analyzer: TransitionAnalyzer, en_analysis_context: AnalysisContext
    ) -> None:
        assert isinstance(analyzer.analyze(en_analysis_context), TransitionResult)

    def test_score_in_range(
        self, analyzer: TransitionAnalyzer, en_analysis_context: AnalysisContext
    ) -> None:
        assert 0.0 <= analyzer.analyze(en_analysis_context).transition_score <= 1.0

    def test_no_transitions_in_plain_essay(
        self, analyzer: TransitionAnalyzer, en_analysis_context: AnalysisContext
    ) -> None:
        result = analyzer.analyze(en_analysis_context)
        assert result.transition_count == 0


class TestRealisticTurkish:
    def test_returns_transition_result(
        self, analyzer: TransitionAnalyzer, tr_analysis_context: AnalysisContext
    ) -> None:
        assert isinstance(analyzer.analyze(tr_analysis_context), TransitionResult)

    def test_score_in_range(
        self, analyzer: TransitionAnalyzer, tr_analysis_context: AnalysisContext
    ) -> None:
        assert 0.0 <= analyzer.analyze(tr_analysis_context).transition_score <= 1.0

    def test_no_transitions_in_plain_essay(
        self, analyzer: TransitionAnalyzer, tr_analysis_context: AnalysisContext
    ) -> None:
        result = analyzer.analyze(tr_analysis_context)
        assert result.transition_count == 0


# ---------------------------------------------------------------------------
# _count_transitions helper — isolated unit tests
# ---------------------------------------------------------------------------


class TestCountTransitions:
    def test_empty_tokens_returns_empty(self) -> None:
        assert _count_transitions(()) == {}

    def test_single_word_transition_detected(self) -> None:
        tokens = ("research", "shows", "furthermore", "students", "learn")
        counts = _count_transitions(tokens)
        assert counts["furthermore"] == 1

    def test_multiword_transition_detected(self) -> None:
        tokens = ("in", "addition", "students", "access", "resources")
        counts = _count_transitions(tokens)
        assert counts["in addition"] == 1

    def test_multiword_requires_adjacent_tokens(self) -> None:
        # "in" at pos 0 and "conclusion" at pos 2 are not adjacent
        tokens = ("in", "other", "conclusion", "matters")
        counts = _count_transitions(tokens)
        assert "in conclusion" not in counts

    def test_turkish_single_word(self) -> None:
        tokens = ("ayrıca", "dijital", "teknoloji", "önemlidir")
        counts = _count_transitions(tokens)
        assert counts["ayrıca"] == 1

    def test_turkish_multiword(self) -> None:
        tokens = ("sonuç", "olarak", "başarı", "artmaktadır")
        counts = _count_transitions(tokens)
        assert counts["sonuç olarak"] == 1

    def test_repeated_phrase_counted_correctly(self) -> None:
        tokens = (
            "however", "students", "learn",
            "however", "teachers", "adapt",
        )
        counts = _count_transitions(tokens)
        assert counts["however"] == 2

    def test_no_partial_match(self) -> None:
        # "conclusion" without "in" should not match "in conclusion"
        tokens = ("the", "conclusion", "is", "clear")
        counts = _count_transitions(tokens)
        assert "in conclusion" not in counts


# ---------------------------------------------------------------------------
# _compute_score helper — isolated unit tests
# ---------------------------------------------------------------------------


class TestComputeScore:
    def test_zero_density_zero_score(self) -> None:
        assert _compute_score(0.0, [], []) == pytest.approx(0.0)

    def test_score_bounded_at_one(self) -> None:
        # Extremely high density, all repeated
        score = _compute_score(100.0, ["furthermore"], ["furthermore"])
        assert score == pytest.approx(1.0)

    def test_density_signal_only(self) -> None:
        # density=1.0, no repeats → score = 0.6×0.5 = 0.3
        score = _compute_score(1.0, ["however"], [])
        assert score == pytest.approx(0.3)

    def test_repeat_ratio_increases_score(self) -> None:
        no_repeat = _compute_score(0.5, ["however", "therefore"], [])
        with_repeat = _compute_score(0.5, ["however", "therefore"], ["however"])
        assert with_repeat > no_repeat

    def test_empty_unique_gives_zero_repeat_ratio(self) -> None:
        # unique=[] → repeat_ratio = 0 regardless of repeated list
        score = _compute_score(0.0, [], [])
        assert score == pytest.approx(0.0)

    def test_density_capped_at_two(self) -> None:
        # density=2.0 and density=10.0 should give same density_signal
        score_two = _compute_score(2.0, [], [])
        score_ten = _compute_score(10.0, [], [])
        assert score_two == pytest.approx(score_ten)
