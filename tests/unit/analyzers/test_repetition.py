"""Unit tests for RepetitionAnalyzer."""

import pytest

from src.analyzers.repetition import (
    RepetitionAnalyzer,
    _all_stop_words,
    _compute_score,
    _find_repeated_openings,
    _find_repeated_phrases,
    _find_repeated_words,
)
from src.models.analysis import AnalysisContext
from src.models.enums import DocumentType, Language
from src.models.response import RepeatedItem, RepetitionResult

# ---------------------------------------------------------------------------
# Module-level AnalysisContext fixtures with pre-verified expected values.
# All arithmetic confirmed independently before writing assertions.
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

# ── _EN_HIGH_WORD_REP ────────────────────────────────────────────────────────
# "technology" (stem "technolog") appears 5 times across 5 sentences,
# every sentence opens with "technology".
#
# Expected:
#   repeated_words  : [technology × 5, positions=[0,5,11,19,26]]
#   repeated_phrases: ["technology has" × 2, positions=[0,5]]
#   repeated_openings: ["technology"]
#   score = 0.5*(4/32) + 0.3*(4/32) + 0.2*(1/5) = 0.14
_EN_HIGH_WORD_REP = AnalysisContext(
    raw_text=(
        "Technology has transformed modern education. "
        "Technology has changed how students learn. "
        "Technology plays a crucial role in academic achievement. "
        "Technology is now essential in every classroom. "
        "Technology enables new forms of collaboration."
    ),
    language=Language.ENGLISH,
    document_type=DocumentType.ACADEMIC,
    cleaned_text="stub",
    tokens=(
        "technology", "has", "transformed", "modern", "education",
        "technology", "has", "changed", "how", "students", "learn",
        "technology", "plays", "a", "crucial", "role", "in", "academic", "achievement",
        "technology", "is", "now", "essential", "in", "every", "classroom",
        "technology", "enables", "new", "forms", "of", "collaboration",
    ),
    stems=(
        "technolog", "has", "transform", "modern", "educ",
        "technolog", "has", "chang", "how", "student", "learn",
        "technolog", "play", "a", "cruci", "role", "in", "academ", "achiev",
        "technolog", "is", "now", "essenti", "in", "everi", "classroom",
        "technolog", "enabl", "new", "form", "of", "collabor",
    ),
    sentences=(
        "Technology has transformed modern education.",
        "Technology has changed how students learn.",
        "Technology plays a crucial role in academic achievement.",
        "Technology is now essential in every classroom.",
        "Technology enables new forms of collaboration.",
    ),
    sentence_token_counts=(5, 6, 8, 7, 6),
)

# ── _EN_PHRASE_REP ───────────────────────────────────────────────────────────
# "research shows that" repeated across 3 sentences → bigrams and trigram repeat.
#
# Expected:
#   repeated_words  : research×3 at [0,6,12], show×3 (surface "shows") at [1,7,13]
#   repeated_phrases: "research shows"×3 at [0,6,12],
#                     "shows that"×3 at [1,7,13],
#                     "research shows that"×3 at [0,6,12]
#   repeated_openings: ["research"]
#   score = 0.5*(4/25) + 0.3*(21/25) + 0.2*(1/4) ≈ 0.382
_EN_PHRASE_REP = AnalysisContext(
    raw_text=(
        "Research shows that learning improves outcomes. "
        "Research shows that technology enhances engagement. "
        "Research shows that digital tools aid comprehension. "
        "Students benefit from evidence based approaches."
    ),
    language=Language.ENGLISH,
    document_type=DocumentType.ACADEMIC,
    cleaned_text="stub",
    tokens=(
        "research", "shows", "that", "learning", "improves", "outcomes",
        "research", "shows", "that", "technology", "enhances", "engagement",
        "research", "shows", "that", "digital", "tools", "aid", "comprehension",
        "students", "benefit", "from", "evidence", "based", "approaches",
    ),
    stems=(
        "research", "show", "that", "learn", "improv", "outcom",
        "research", "show", "that", "technolog", "enhanc", "engag",
        "research", "show", "that", "digit", "tool", "aid", "comprehens",
        "student", "benefit", "from", "evid", "base", "approach",
    ),
    sentences=(
        "Research shows that learning improves outcomes.",
        "Research shows that technology enhances engagement.",
        "Research shows that digital tools aid comprehension.",
        "Students benefit from evidence based approaches.",
    ),
    sentence_token_counts=(6, 6, 7, 6),
)

# ── _EN_VARIED ───────────────────────────────────────────────────────────────
# Diverse academic vocabulary — every content stem appears exactly once,
# single sentence so no repeated openers.
#
# Expected: repeated_words=[], repeated_phrases=[], repeated_openings=[], score=0.0
_EN_VARIED = AnalysisContext(
    raw_text="Language shapes perception reality philosophers debate structure constrains thought.",
    language=Language.ENGLISH,
    document_type=DocumentType.ESSAY,
    cleaned_text="stub",
    tokens=(
        "language", "shapes", "perception", "reality", "philosophers",
        "debate", "structure", "constrains", "thought", "researchers",
        "argue", "categories", "influence", "cognition", "evidence",
        "demonstrates", "relationship", "approaches",
    ),
    stems=(
        "languag", "shape", "percept", "realiti", "philosoph",
        "debat", "structur", "constrain", "thought", "research",
        "argu", "categori", "influenc", "cognit", "evid",
        "demonstr", "relat", "approach",
    ),
    sentences=("Language shapes perception reality philosophers debate structure.",),
    sentence_token_counts=(18,),
)

# ── _EN_BELOW_THRESHOLD ──────────────────────────────────────────────────────
# "research" appears only twice (below word_min_count=3) and sentence
# openers differ — no signals should be raised.
#
# Expected: repeated_words=[], repeated_phrases=[], repeated_openings=[], score=0.0
_EN_BELOW_THRESHOLD = AnalysisContext(
    raw_text="Research confirms the hypothesis. Evidence supports the conclusion.",
    language=Language.ENGLISH,
    document_type=DocumentType.ESSAY,
    cleaned_text="stub",
    tokens=("research", "confirms", "the", "hypothesis", "evidence", "supports", "the", "conclusion"),
    stems=("research", "confirm", "the", "hypothesi", "evid", "support", "the", "conclus"),
    sentences=(
        "Research confirms the hypothesis.",
        "Evidence supports the conclusion.",
    ),
    sentence_token_counts=(4, 4),
)

# ── _EN_STOP_OPENERS ─────────────────────────────────────────────────────────
# Every sentence starts with "the" — a stop word.
# Stop-word filtering must prevent "the" from appearing in repeated_openings.
#
# Expected: repeated_openings=[]
_EN_STOP_OPENERS = AnalysisContext(
    raw_text="The cat slept. The dog ran. The bird flew.",
    language=Language.ENGLISH,
    document_type=DocumentType.ESSAY,
    cleaned_text="stub",
    tokens=("the", "cat", "slept", "the", "dog", "ran", "the", "bird", "flew"),
    stems=("the", "cat", "sleep", "the", "dog", "run", "the", "bird", "fli"),
    sentences=("The cat slept.", "The dog ran.", "The bird flew."),
    sentence_token_counts=(3, 3, 3),
)

# ── _TR_REPETITIVE ───────────────────────────────────────────────────────────
# Turkish text: "eğitim" (education) appears 4 times across 4 sentences.
# Every sentence opens with "eğitim".
#
# Expected:
#   repeated_words  : [eğitim × 4, positions=[0,4,7,11]]
#   repeated_phrases: []
#   repeated_openings: ["eğitim"]
#   score = 0.5*(3/15) + 0.3*0 + 0.2*(1/4) = 0.15
_TR_REPETITIVE = AnalysisContext(
    raw_text=(
        "Eğitim dijital araçlarla dönüşmektedir. "
        "Eğitim kalitesi artmaktadır. "
        "Eğitim kurumları teknolojiyi benimsemektedir. "
        "Eğitim sürecinde teknoloji önemlidir."
    ),
    language=Language.TURKISH,
    document_type=DocumentType.ACADEMIC,
    cleaned_text="stub",
    tokens=(
        "eğitim", "dijital", "araçlarla", "dönüşmektedir",
        "eğitim", "kalitesi", "artmaktadır",
        "eğitim", "kurumları", "teknolojiyi", "benimsemektedir",
        "eğitim", "sürecinde", "teknoloji", "önemlidir",
    ),
    stems=(
        "eğitim", "dijital", "araç", "dönüştür",
        "eğitim", "kalite", "artır",
        "eğitim", "kurum", "teknoloji", "benimse",
        "eğitim", "süreç", "teknoloji", "önem",
    ),
    sentences=(
        "Eğitim dijital araçlarla dönüşmektedir.",
        "Eğitim kalitesi artmaktadır.",
        "Eğitim kurumları teknolojiyi benimsemektedir.",
        "Eğitim sürecinde teknoloji önemlidir.",
    ),
    sentence_token_counts=(4, 3, 4, 4),
)


# ---------------------------------------------------------------------------
# Shared analyzer instance — RepetitionAnalyzer is stateless
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analyzer() -> RepetitionAnalyzer:
    """Shared RepetitionAnalyzer with default thresholds."""
    return RepetitionAnalyzer()


# ---------------------------------------------------------------------------
# Helpers for assertions that don't depend on list ordering
# ---------------------------------------------------------------------------


def _word_texts(result: RepetitionResult) -> set[str]:
    return {item.text for item in result.repeated_words}


def _phrase_texts(result: RepetitionResult) -> set[str]:
    return {item.text for item in result.repeated_phrases}


def _word_by_text(result: RepetitionResult, text: str) -> RepeatedItem:
    return next(item for item in result.repeated_words if item.text == text)


def _phrase_by_text(result: RepetitionResult, text: str) -> RepeatedItem:
    return next(item for item in result.repeated_phrases if item.text == text)


# ---------------------------------------------------------------------------
# Analyzer identity and construction
# ---------------------------------------------------------------------------


class TestAnalyzerIdentity:
    def test_name_is_repetition(self, analyzer: RepetitionAnalyzer):
        assert analyzer.name == "repetition"

    def test_analyze_returns_repetition_result(self, analyzer: RepetitionAnalyzer):
        assert isinstance(analyzer.analyze(_EN_VARIED), RepetitionResult)

    def test_default_word_min_count(self):
        a = RepetitionAnalyzer()
        assert a._word_min == 3

    def test_default_phrase_min_count(self):
        assert RepetitionAnalyzer()._phrase_min == 2

    def test_default_opening_min_count(self):
        assert RepetitionAnalyzer()._opening_min == 2

    def test_custom_thresholds_accepted(self):
        a = RepetitionAnalyzer(word_min_count=4, phrase_min_count=3, opening_min_count=3)
        assert a._word_min == 4 and a._phrase_min == 3 and a._opening_min == 3


class TestParameterValidation:
    def test_word_min_below_two_raises(self):
        with pytest.raises(ValueError, match="word_min_count"):
            RepetitionAnalyzer(word_min_count=1)

    def test_phrase_min_below_two_raises(self):
        with pytest.raises(ValueError, match="phrase_min_count"):
            RepetitionAnalyzer(phrase_min_count=1)

    def test_opening_min_below_two_raises(self):
        with pytest.raises(ValueError, match="opening_min_count"):
            RepetitionAnalyzer(opening_min_count=1)

    def test_zero_word_min_raises(self):
        with pytest.raises(ValueError):
            RepetitionAnalyzer(word_min_count=0)


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_repeated_words_empty(self, analyzer: RepetitionAnalyzer):
        assert analyzer.analyze(_EMPTY).repeated_words == []

    def test_repeated_phrases_empty(self, analyzer: RepetitionAnalyzer):
        assert analyzer.analyze(_EMPTY).repeated_phrases == []

    def test_repeated_openings_empty(self, analyzer: RepetitionAnalyzer):
        assert analyzer.analyze(_EMPTY).repeated_openings == []

    def test_score_is_zero(self, analyzer: RepetitionAnalyzer):
        assert analyzer.analyze(_EMPTY).repetition_score == 0.0

    def test_returns_repetition_result(self, analyzer: RepetitionAnalyzer):
        assert isinstance(analyzer.analyze(_EMPTY), RepetitionResult)


# ---------------------------------------------------------------------------
# Below-threshold text — nothing flagged, score = 0
# ---------------------------------------------------------------------------


class TestBelowThreshold:
    def test_no_repeated_words(self, analyzer: RepetitionAnalyzer):
        assert analyzer.analyze(_EN_BELOW_THRESHOLD).repeated_words == []

    def test_no_repeated_phrases(self, analyzer: RepetitionAnalyzer):
        assert analyzer.analyze(_EN_BELOW_THRESHOLD).repeated_phrases == []

    def test_no_repeated_openings(self, analyzer: RepetitionAnalyzer):
        assert analyzer.analyze(_EN_BELOW_THRESHOLD).repeated_openings == []

    def test_score_is_zero(self, analyzer: RepetitionAnalyzer):
        assert analyzer.analyze(_EN_BELOW_THRESHOLD).repetition_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Stop-word filtering
# ---------------------------------------------------------------------------


class TestStopWordFiltering:
    def test_stop_word_opener_not_flagged(self, analyzer: RepetitionAnalyzer):
        # "the" starts every sentence but must not appear in repeated_openings
        result = analyzer.analyze(_EN_STOP_OPENERS)
        assert "the" not in result.repeated_openings

    def test_repeated_openings_empty_for_stop_openers(self, analyzer: RepetitionAnalyzer):
        assert analyzer.analyze(_EN_STOP_OPENERS).repeated_openings == []

    def test_score_zero_for_stop_opener_only_text(self, analyzer: RepetitionAnalyzer):
        # "cat", "dog", "bird" each appear once; "the" is stop — no signal
        assert analyzer.analyze(_EN_STOP_OPENERS).repetition_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Varied English text — no signals raised
# ---------------------------------------------------------------------------


class TestVariedText:
    def test_no_repeated_words(self, analyzer: RepetitionAnalyzer):
        assert analyzer.analyze(_EN_VARIED).repeated_words == []

    def test_no_repeated_phrases(self, analyzer: RepetitionAnalyzer):
        assert analyzer.analyze(_EN_VARIED).repeated_phrases == []

    def test_no_repeated_openings(self, analyzer: RepetitionAnalyzer):
        assert analyzer.analyze(_EN_VARIED).repeated_openings == []

    def test_score_is_zero(self, analyzer: RepetitionAnalyzer):
        assert analyzer.analyze(_EN_VARIED).repetition_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# High word repetition — "technology" × 5
# ---------------------------------------------------------------------------


class TestHighWordRepetition:
    def test_technology_detected(self, analyzer: RepetitionAnalyzer):
        assert "technology" in _word_texts(analyzer.analyze(_EN_HIGH_WORD_REP))

    def test_technology_count(self, analyzer: RepetitionAnalyzer):
        item = _word_by_text(analyzer.analyze(_EN_HIGH_WORD_REP), "technology")
        assert item.count == 5

    def test_technology_positions(self, analyzer: RepetitionAnalyzer):
        item = _word_by_text(analyzer.analyze(_EN_HIGH_WORD_REP), "technology")
        assert item.positions == [0, 5, 11, 19, 26]

    def test_single_word_flagged(self, analyzer: RepetitionAnalyzer):
        # Only "technology" meets the threshold; all others are stop words or appear ≤ 2×
        assert len(analyzer.analyze(_EN_HIGH_WORD_REP).repeated_words) == 1

    def test_repeated_words_sorted_descending(self, analyzer: RepetitionAnalyzer):
        words = analyzer.analyze(_EN_HIGH_WORD_REP).repeated_words
        counts = [w.count for w in words]
        assert counts == sorted(counts, reverse=True)

    def test_technology_bigram_detected(self, analyzer: RepetitionAnalyzer):
        assert "technology has" in _phrase_texts(analyzer.analyze(_EN_HIGH_WORD_REP))

    def test_technology_bigram_count(self, analyzer: RepetitionAnalyzer):
        item = _phrase_by_text(analyzer.analyze(_EN_HIGH_WORD_REP), "technology has")
        assert item.count == 2

    def test_technology_bigram_positions(self, analyzer: RepetitionAnalyzer):
        item = _phrase_by_text(analyzer.analyze(_EN_HIGH_WORD_REP), "technology has")
        assert item.positions == [0, 5]

    def test_technology_is_repeated_opener(self, analyzer: RepetitionAnalyzer):
        assert "technology" in analyzer.analyze(_EN_HIGH_WORD_REP).repeated_openings

    def test_score_exact(self, analyzer: RepetitionAnalyzer):
        # word=(4/32), phrase=(4/32), opener=1/5 → 0.5*0.125+0.3*0.125+0.2*0.2 = 0.14
        assert analyzer.analyze(_EN_HIGH_WORD_REP).repetition_score == pytest.approx(0.14)

    def test_score_higher_than_varied_text(self, analyzer: RepetitionAnalyzer):
        high = analyzer.analyze(_EN_HIGH_WORD_REP).repetition_score
        low = analyzer.analyze(_EN_VARIED).repetition_score
        assert high > low


# ---------------------------------------------------------------------------
# Phrase repetition — "research shows that" × 3
# ---------------------------------------------------------------------------


class TestPhraseRepetition:
    def test_research_shows_bigram_detected(self, analyzer: RepetitionAnalyzer):
        assert "research shows" in _phrase_texts(analyzer.analyze(_EN_PHRASE_REP))

    def test_shows_that_bigram_detected(self, analyzer: RepetitionAnalyzer):
        assert "shows that" in _phrase_texts(analyzer.analyze(_EN_PHRASE_REP))

    def test_research_shows_that_trigram_detected(self, analyzer: RepetitionAnalyzer):
        assert "research shows that" in _phrase_texts(analyzer.analyze(_EN_PHRASE_REP))

    def test_research_shows_count(self, analyzer: RepetitionAnalyzer):
        assert _phrase_by_text(analyzer.analyze(_EN_PHRASE_REP), "research shows").count == 3

    def test_research_shows_that_count(self, analyzer: RepetitionAnalyzer):
        assert _phrase_by_text(analyzer.analyze(_EN_PHRASE_REP), "research shows that").count == 3

    def test_research_shows_positions(self, analyzer: RepetitionAnalyzer):
        item = _phrase_by_text(analyzer.analyze(_EN_PHRASE_REP), "research shows")
        assert item.positions == [0, 6, 12]

    def test_shows_that_positions(self, analyzer: RepetitionAnalyzer):
        item = _phrase_by_text(analyzer.analyze(_EN_PHRASE_REP), "shows that")
        assert item.positions == [1, 7, 13]

    def test_research_word_detected(self, analyzer: RepetitionAnalyzer):
        assert "research" in _word_texts(analyzer.analyze(_EN_PHRASE_REP))

    def test_research_word_count(self, analyzer: RepetitionAnalyzer):
        assert _word_by_text(analyzer.analyze(_EN_PHRASE_REP), "research").count == 3

    def test_shows_word_detected(self, analyzer: RepetitionAnalyzer):
        # surface "shows" maps to stem "show"; display text = most common surface form
        assert "shows" in _word_texts(analyzer.analyze(_EN_PHRASE_REP))

    def test_shows_word_positions(self, analyzer: RepetitionAnalyzer):
        item = _word_by_text(analyzer.analyze(_EN_PHRASE_REP), "shows")
        assert item.positions == [1, 7, 13]

    def test_research_is_repeated_opener(self, analyzer: RepetitionAnalyzer):
        assert "research" in analyzer.analyze(_EN_PHRASE_REP).repeated_openings

    def test_score_exact(self, analyzer: RepetitionAnalyzer):
        # 0.5*(4/25) + 0.3*(21/25) + 0.2*(1/4) = 0.382
        assert analyzer.analyze(_EN_PHRASE_REP).repetition_score == pytest.approx(0.382)

    def test_score_higher_than_word_only_repetition(self, analyzer: RepetitionAnalyzer):
        # phrase repetition generates a higher score than word-only repetition
        phrase_score = analyzer.analyze(_EN_PHRASE_REP).repetition_score
        word_score = analyzer.analyze(_EN_HIGH_WORD_REP).repetition_score
        assert phrase_score > word_score


# ---------------------------------------------------------------------------
# Turkish repetition
# ---------------------------------------------------------------------------


class TestTurkishRepetition:
    def test_egitim_detected(self, analyzer: RepetitionAnalyzer):
        assert "eğitim" in _word_texts(analyzer.analyze(_TR_REPETITIVE))

    def test_egitim_count(self, analyzer: RepetitionAnalyzer):
        item = _word_by_text(analyzer.analyze(_TR_REPETITIVE), "eğitim")
        assert item.count == 4

    def test_egitim_positions(self, analyzer: RepetitionAnalyzer):
        item = _word_by_text(analyzer.analyze(_TR_REPETITIVE), "eğitim")
        assert item.positions == [0, 4, 7, 11]

    def test_no_repeated_phrases(self, analyzer: RepetitionAnalyzer):
        assert analyzer.analyze(_TR_REPETITIVE).repeated_phrases == []

    def test_egitim_is_repeated_opener(self, analyzer: RepetitionAnalyzer):
        assert "eğitim" in analyzer.analyze(_TR_REPETITIVE).repeated_openings

    def test_score_exact(self, analyzer: RepetitionAnalyzer):
        # 0.5*(3/15) + 0 + 0.2*(1/4) = 0.15
        assert analyzer.analyze(_TR_REPETITIVE).repetition_score == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Structural invariants across all contexts
# ---------------------------------------------------------------------------


class TestInvariants:
    @pytest.mark.parametrize("ctx", [
        _EMPTY, _EN_HIGH_WORD_REP, _EN_PHRASE_REP,
        _EN_VARIED, _EN_BELOW_THRESHOLD, _EN_STOP_OPENERS, _TR_REPETITIVE,
    ])
    def test_score_in_unit_interval(
        self, analyzer: RepetitionAnalyzer, ctx: AnalysisContext
    ):
        score = analyzer.analyze(ctx).repetition_score
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("ctx", [
        _EMPTY, _EN_HIGH_WORD_REP, _EN_PHRASE_REP,
        _EN_VARIED, _EN_BELOW_THRESHOLD, _EN_STOP_OPENERS, _TR_REPETITIVE,
    ])
    def test_repeated_words_count_at_least_two(
        self, analyzer: RepetitionAnalyzer, ctx: AnalysisContext
    ):
        for item in analyzer.analyze(ctx).repeated_words:
            assert item.count >= 2

    @pytest.mark.parametrize("ctx", [
        _EMPTY, _EN_HIGH_WORD_REP, _EN_PHRASE_REP,
        _EN_VARIED, _EN_BELOW_THRESHOLD, _EN_STOP_OPENERS, _TR_REPETITIVE,
    ])
    def test_repeated_phrases_count_at_least_two(
        self, analyzer: RepetitionAnalyzer, ctx: AnalysisContext
    ):
        for item in analyzer.analyze(ctx).repeated_phrases:
            assert item.count >= 2

    @pytest.mark.parametrize("ctx", [
        _EMPTY, _EN_HIGH_WORD_REP, _EN_PHRASE_REP,
        _EN_VARIED, _EN_BELOW_THRESHOLD, _EN_STOP_OPENERS, _TR_REPETITIVE,
    ])
    def test_positions_length_matches_count(
        self, analyzer: RepetitionAnalyzer, ctx: AnalysisContext
    ):
        result = analyzer.analyze(ctx)
        for item in result.repeated_words + result.repeated_phrases:
            assert len(item.positions) == item.count

    @pytest.mark.parametrize("ctx", [
        _EMPTY, _EN_HIGH_WORD_REP, _EN_PHRASE_REP,
        _EN_VARIED, _EN_BELOW_THRESHOLD, _EN_STOP_OPENERS, _TR_REPETITIVE,
    ])
    def test_pydantic_model_validates(
        self, analyzer: RepetitionAnalyzer, ctx: AnalysisContext
    ):
        result = analyzer.analyze(ctx)
        assert RepetitionResult.model_validate(result.model_dump())

    @pytest.mark.parametrize("ctx", [
        _EMPTY, _EN_HIGH_WORD_REP, _EN_PHRASE_REP,
        _EN_VARIED, _EN_BELOW_THRESHOLD, _EN_STOP_OPENERS, _TR_REPETITIVE,
    ])
    def test_positions_are_sorted_ascending(
        self, analyzer: RepetitionAnalyzer, ctx: AnalysisContext
    ):
        result = analyzer.analyze(ctx)
        for item in result.repeated_words + result.repeated_phrases:
            assert item.positions == sorted(item.positions)


# ---------------------------------------------------------------------------
# Realistic conftest fixtures
# ---------------------------------------------------------------------------


class TestRealisticEnglish:
    def test_returns_repetition_result(
        self, analyzer: RepetitionAnalyzer, en_analysis_context: AnalysisContext
    ):
        assert isinstance(analyzer.analyze(en_analysis_context), RepetitionResult)

    def test_score_in_range(
        self, analyzer: RepetitionAnalyzer, en_analysis_context: AnalysisContext
    ):
        score = analyzer.analyze(en_analysis_context).repetition_score
        assert 0.0 <= score <= 1.0

    def test_repeated_word_counts_satisfy_constraint(
        self, analyzer: RepetitionAnalyzer, en_analysis_context: AnalysisContext
    ):
        for item in analyzer.analyze(en_analysis_context).repeated_words:
            assert item.count >= 2


class TestRealisticTurkish:
    def test_returns_repetition_result(
        self, analyzer: RepetitionAnalyzer, tr_analysis_context: AnalysisContext
    ):
        assert isinstance(analyzer.analyze(tr_analysis_context), RepetitionResult)

    def test_score_in_range(
        self, analyzer: RepetitionAnalyzer, tr_analysis_context: AnalysisContext
    ):
        score = analyzer.analyze(tr_analysis_context).repetition_score
        assert 0.0 <= score <= 1.0

    def test_repeated_phrases_count_constraint(
        self, analyzer: RepetitionAnalyzer, tr_analysis_context: AnalysisContext
    ):
        for item in analyzer.analyze(tr_analysis_context).repeated_phrases:
            assert item.count >= 2


# ---------------------------------------------------------------------------
# Custom threshold behaviour
# ---------------------------------------------------------------------------


class TestCustomThreshold:
    def test_higher_word_threshold_reduces_detections(self):
        strict = RepetitionAnalyzer(word_min_count=6)
        # "technology" appears 5 times — below min of 6 → not flagged
        result = strict.analyze(_EN_HIGH_WORD_REP)
        assert "technology" not in _word_texts(result)

    def test_higher_phrase_threshold_reduces_detections(self):
        strict = RepetitionAnalyzer(phrase_min_count=4)
        # "research shows" appears 3 times — below min of 4 → not flagged
        result = strict.analyze(_EN_PHRASE_REP)
        assert "research shows" not in _phrase_texts(result)

    def test_higher_threshold_lowers_score(self):
        default = RepetitionAnalyzer()
        strict = RepetitionAnalyzer(word_min_count=10, phrase_min_count=10, opening_min_count=10)
        assert default.analyze(_EN_HIGH_WORD_REP).repetition_score >= \
               strict.analyze(_EN_HIGH_WORD_REP).repetition_score


# ---------------------------------------------------------------------------
# Pure helper functions — unit-tested in isolation
# ---------------------------------------------------------------------------


class TestAllStopWords:
    def test_all_stop_returns_true(self):
        assert _all_stop_words("the", "and", "of") is True

    def test_content_word_returns_false(self):
        assert _all_stop_words("the", "research") is False

    def test_single_stop_word(self):
        assert _all_stop_words("the") is True

    def test_single_content_word(self):
        assert _all_stop_words("technology") is False

    def test_turkish_stop_words(self):
        assert _all_stop_words("ve", "bir") is True

    def test_mixed_language_stop_words(self):
        assert _all_stop_words("the", "ve") is True


class TestFindRepeatedWords:
    def test_returns_empty_for_empty_tokens(self):
        assert _find_repeated_words((), (), 3) == []

    def test_detects_stem_above_threshold(self):
        tokens = ("run", "runs", "running", "walk")
        stems = ("run", "run", "run", "walk")
        items = _find_repeated_words(tokens, stems, 3)
        assert len(items) == 1
        assert items[0].count == 3

    def test_most_common_surface_form_chosen(self):
        # "run" appears twice, "running" once → display should be "run"
        tokens = ("run", "run", "running")
        stems = ("run", "run", "run")
        items = _find_repeated_words(tokens, stems, 3)
        assert items[0].text == "run"

    def test_stop_words_excluded(self):
        tokens = ("the", "the", "the", "cat")
        stems = ("the", "the", "the", "cat")
        assert _find_repeated_words(tokens, stems, 3) == []

    def test_below_threshold_excluded(self):
        tokens = ("study", "study", "analysis")
        stems = ("studi", "studi", "analysi")
        assert _find_repeated_words(tokens, stems, 3) == []


class TestFindRepeatedPhrases:
    def test_returns_empty_for_empty_tokens(self):
        assert _find_repeated_phrases((), 2) == []

    def test_detects_repeated_bigram(self):
        tokens = ("research", "shows", "that", "research", "shows", "results")
        items = _find_repeated_phrases(tokens, 2)
        texts = {i.text for i in items}
        assert "research shows" in texts

    def test_excludes_all_stop_word_bigram(self):
        tokens = ("of", "the", "study", "of", "the", "data")
        items = _find_repeated_phrases(tokens, 2)
        texts = {i.text for i in items}
        assert "of the" not in texts

    def test_single_token_no_phrases(self):
        assert _find_repeated_phrases(("word",), 2) == []

    def test_positions_are_start_indices(self):
        tokens = ("a", "b", "c", "a", "b", "d")
        items = _find_repeated_phrases(tokens, 2)
        ab = next(i for i in items if i.text == "a b")
        assert ab.positions == [0, 3]


class TestFindRepeatedOpenings:
    def test_returns_empty_for_empty_context(self):
        assert _find_repeated_openings((), (), 2) == []

    def test_detects_repeated_opener(self):
        tokens = ("research", "confirms", "research", "shows")
        counts = (2, 2)
        result = _find_repeated_openings(tokens, counts, 2)
        assert "research" in result

    def test_stop_word_opener_excluded(self):
        tokens = ("the", "cat", "the", "dog")
        counts = (2, 2)
        assert _find_repeated_openings(tokens, counts, 2) == []

    def test_below_threshold_not_flagged(self):
        tokens = ("research", "confirms", "evidence", "shows")
        counts = (2, 2)
        assert _find_repeated_openings(tokens, counts, 3) == []


class TestComputeScore:
    def test_zero_tokens_returns_zero(self):
        assert _compute_score([], [], [], 0, 0) == 0.0

    def test_no_repetitions_returns_zero(self):
        assert _compute_score([], [], [], 100, 10) == pytest.approx(0.0)

    def test_score_bounded_at_one(self):
        # Artificially large repeated_words excess
        items = [RepeatedItem(text="word", count=1000, positions=list(range(1000)))]
        score = _compute_score(items, [], [], 100, 10)
        assert score == pytest.approx(1.0)

    def test_word_signal_only(self):
        # excess = 4 tokens out of 32, weight 0.5 → 0.5*(4/32) = 0.0625
        items = [RepeatedItem(text="technology", count=5, positions=[0,5,11,19,26])]
        score = _compute_score(items, [], [], 32, 5)
        assert score == pytest.approx(0.0625)
