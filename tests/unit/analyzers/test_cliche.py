"""Unit tests for ClicheAnalyzer."""

import pytest

from src.analyzers.cliche import (
    _ALL_CLICHES,
    _EN_CLICHES,
    _TR_CLICHES,
    ClicheAnalyzer,
    _count_cliches,
)
from src.models.analysis import AnalysisContext
from src.models.enums import DocumentType, Language
from src.models.response import ClicheResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _en_ctx(tokens: tuple[str, ...]) -> AnalysisContext:
    """Build a minimal English AnalysisContext from a token tuple."""
    n = len(tokens)
    return AnalysisContext(
        raw_text="stub",
        language=Language.ENGLISH,
        document_type=DocumentType.ESSAY,
        cleaned_text="stub",
        tokens=tokens,
        sentences=("stub",) if n else (),
        stems=tokens,
        sentence_token_counts=(n,) if n else (),
    )


def _tr_ctx(tokens: tuple[str, ...]) -> AnalysisContext:
    """Build a minimal Turkish AnalysisContext from a token tuple."""
    n = len(tokens)
    return AnalysisContext(
        raw_text="stub",
        language=Language.TURKISH,
        document_type=DocumentType.ESSAY,
        cleaned_text="stub",
        tokens=tokens,
        sentences=("stub",) if n else (),
        stems=tokens,
        sentence_token_counts=(n,) if n else (),
    )


# ---------------------------------------------------------------------------
# Module-level AnalysisContext fixtures with pre-verified expected values.
#
# All fixtures use 100 tokens so that density = cliche_count × 1.0.
# score = min(1.0, density / 5.0)
# ---------------------------------------------------------------------------

# ── _EMPTY ───────────────────────────────────────────────────────────────────
_EMPTY = _en_ctx(())

# ── _EN_NO_CLICHES ────────────────────────────────────────────────────────────
# 8 tokens — no token sequence matches any registered phrase.
# Expected: detected_cliches=[], count=0, density=0.0, score=0.0
_EN_NO_CLICHES = _en_ctx(
    ("language", "shapes", "perception", "philosophers", "debate", "its", "nature", "deeply")
)

# ── _EN_SINGLE ────────────────────────────────────────────────────────────────
# 100 tokens, "in conclusion" once at positions [20,21].
# count=1, density=1.0, score=0.2
_EN_SINGLE = _en_ctx(
    ("word",) * 20 + ("in", "conclusion") + ("word",) * 78
)

# ── _EN_MULTI ─────────────────────────────────────────────────────────────────
# 100 tokens, "in conclusion" ×2 + "needless to say" ×1.
# count=3, density=3.0, score=0.6
# detected_cliches=["in conclusion", "needless to say"]
_EN_MULTI = _en_ctx(
    ("word",) * 20
    + ("in", "conclusion")
    + ("word",) * 25
    + ("in", "conclusion")
    + ("word",) * 15
    + ("needless", "to", "say")
    + ("word",) * 33
)

# ── _EN_SATURATED ─────────────────────────────────────────────────────────────
# 100 tokens, five different clichés once each (NLTK apostrophe variant).
# count=5, density=5.0, score=1.0
# detected_cliches (sorted):
#   ["in conclusion", "in today's world", "it goes without saying",
#    "needless to say", "the fact of the matter is"]
_EN_SATURATED = _en_ctx(
    ("word",) * 10
    + ("in", "conclusion")
    + ("word",) * 17
    + ("needless", "to", "say")
    + ("word",) * 17
    + ("it", "goes", "without", "saying")
    + ("word",) * 17
    + ("in", "today", "world")          # NLTK variant of "in today's world"
    + ("word",) * 17
    + ("the", "fact", "of", "the", "matter", "is")
    + ("word",) * 4
)

# ── _EN_LONG_PHRASE ───────────────────────────────────────────────────────────
# 100 tokens, "it is important to note that" (6 tokens) once.
# count=1, density=1.0, score=0.2
_EN_LONG_PHRASE = _en_ctx(
    ("word",) * 30
    + ("it", "is", "important", "to", "note", "that")
    + ("word",) * 64
)

# ── _EN_APOSTROPHE_REGEX ──────────────────────────────────────────────────────
# 100 tokens, "in today's world" in regex tokenizer form ("s" is a separate token).
# count=1, density=1.0, score=0.2
_EN_APOSTROPHE_REGEX = _en_ctx(
    ("word",) * 20 + ("in", "today", "s", "world") + ("word",) * 76
)

# ── _TR_TWO_CLICHES ──────────────────────────────────────────────────────────
# 100 tokens, "sonuç olarak" ×1 + "günümüzde" ×1.
# count=2, density=2.0, score=0.4
# detected_cliches=["günümüzde", "sonuç olarak"]
_TR_TWO_CLICHES = _tr_ctx(
    ("sonuç", "olarak")
    + ("kelime",) * 50
    + ("günümüzde",)
    + ("kelime",) * 47
)

# ── _TR_FOUR_CLICHES ─────────────────────────────────────────────────────────
# 100 tokens, four Turkish clichés once each.
# count=4, density=4.0, score=0.8
# detected_cliches (sorted Unicode):
#   ["bilindiği üzere", "büyük önem taşımaktadır", "günümüzde", "sonuç olarak"]
_TR_FOUR_CLICHES = _tr_ctx(
    ("bilindiği", "üzere")
    + ("kelime",) * 20
    + ("büyük", "önem", "taşımaktadır")
    + ("kelime",) * 20
    + ("sonuç", "olarak")
    + ("kelime",) * 20
    + ("günümüzde",)
    + ("kelime",) * 32
)

# Fixture bundle for parametrized tests
_ALL_FIXTURES = [
    _EMPTY,
    _EN_NO_CLICHES,
    _EN_SINGLE,
    _EN_MULTI,
    _EN_SATURATED,
    _EN_LONG_PHRASE,
    _EN_APOSTROPHE_REGEX,
    _TR_TWO_CLICHES,
    _TR_FOUR_CLICHES,
]


# ===========================================================================
# Test classes
# ===========================================================================


class TestAnalyzerIdentity:
    """Static properties of the analyzer object."""

    def test_name(self) -> None:
        assert ClicheAnalyzer().name == "cliche"

    def test_analyze_returns_cliche_result(self) -> None:
        assert isinstance(ClicheAnalyzer().analyze(_EMPTY), ClicheResult)

    def test_two_instances_are_independent(self) -> None:
        a, b = ClicheAnalyzer(), ClicheAnalyzer()
        assert a.analyze(_EN_SINGLE) == b.analyze(_EN_SINGLE)


class TestRegisteredPhrases:
    """Verify the canonical phrase registries are complete and consistent."""

    def test_en_cliches_count(self) -> None:
        assert len(_EN_CLICHES) == 6

    def test_tr_cliches_count(self) -> None:
        assert len(_TR_CLICHES) == 6

    def test_all_cliches_is_concatenation(self) -> None:
        assert _ALL_CLICHES == _EN_CLICHES + _TR_CLICHES

    def test_en_cliches_are_lowercase(self) -> None:
        for phrase in _EN_CLICHES:
            assert phrase == phrase.lower(), f"Not lowercase: {phrase!r}"

    def test_tr_cliches_are_lowercase(self) -> None:
        for phrase in _TR_CLICHES:
            assert phrase == phrase.lower(), f"Not lowercase: {phrase!r}"

    def test_specific_en_phrases_present(self) -> None:
        assert "in conclusion" in _EN_CLICHES
        assert "it is important to note that" in _EN_CLICHES
        assert "in today's world" in _EN_CLICHES
        assert "needless to say" in _EN_CLICHES
        assert "the fact of the matter is" in _EN_CLICHES
        assert "it goes without saying" in _EN_CLICHES

    def test_specific_tr_phrases_present(self) -> None:
        assert "sonuç olarak" in _TR_CLICHES
        assert "günümüzde" in _TR_CLICHES
        assert "bilindiği üzere" in _TR_CLICHES
        assert "büyük önem taşımaktadır" in _TR_CLICHES
        assert "yadsınamaz bir gerçektir ki" in _TR_CLICHES
        assert "bunun altını çizmek gerekir" in _TR_CLICHES


class TestEmptyInput:
    """Empty context returns zero-valued default."""

    def setup_method(self) -> None:
        self._result = ClicheAnalyzer().analyze(_EMPTY)

    def test_detected_cliches_empty(self) -> None:
        assert self._result.detected_cliches == []

    def test_cliche_count_zero(self) -> None:
        assert self._result.cliche_count == 0

    def test_density_zero(self) -> None:
        assert self._result.cliche_density == 0.0

    def test_score_zero(self) -> None:
        assert self._result.cliche_score == 0.0


class TestNoClichesEnglish:
    """Tokens with no matching phrases return empty results."""

    def setup_method(self) -> None:
        self._result = ClicheAnalyzer().analyze(_EN_NO_CLICHES)

    def test_detected_cliches_empty(self) -> None:
        assert self._result.detected_cliches == []

    def test_cliche_count_zero(self) -> None:
        assert self._result.cliche_count == 0

    def test_density_zero(self) -> None:
        assert self._result.cliche_density == 0.0

    def test_score_zero(self) -> None:
        assert self._result.cliche_score == 0.0


class TestSingleEnglishCliche:
    """100-token text with one 'in conclusion' occurrence."""

    def setup_method(self) -> None:
        self._result = ClicheAnalyzer().analyze(_EN_SINGLE)

    def test_detected_cliches(self) -> None:
        assert self._result.detected_cliches == ["in conclusion"]

    def test_cliche_count(self) -> None:
        assert self._result.cliche_count == 1

    def test_density(self) -> None:
        assert self._result.cliche_density == pytest.approx(1.0)

    def test_score(self) -> None:
        assert self._result.cliche_score == pytest.approx(0.2)


class TestMultipleEnglishCliches:
    """100-token text with 'in conclusion' ×2 and 'needless to say' ×1."""

    def setup_method(self) -> None:
        self._result = ClicheAnalyzer().analyze(_EN_MULTI)

    def test_detected_cliches(self) -> None:
        # unique sorted phrases — "in conclusion" appears twice but listed once
        assert self._result.detected_cliches == ["in conclusion", "needless to say"]

    def test_cliche_count(self) -> None:
        assert self._result.cliche_count == 3

    def test_density(self) -> None:
        assert self._result.cliche_density == pytest.approx(3.0)

    def test_score(self) -> None:
        assert self._result.cliche_score == pytest.approx(0.6)

    def test_detected_cliches_deduplicated(self) -> None:
        # "in conclusion" occurs twice but detected_cliches lists it once
        assert self._result.detected_cliches.count("in conclusion") == 1


class TestSaturatedScore:
    """Five different clichés yield the maximum score of 1.0."""

    def setup_method(self) -> None:
        self._result = ClicheAnalyzer().analyze(_EN_SATURATED)

    def test_cliche_count(self) -> None:
        assert self._result.cliche_count == 5

    def test_density(self) -> None:
        assert self._result.cliche_density == pytest.approx(5.0)

    def test_score_capped_at_one(self) -> None:
        assert self._result.cliche_score == pytest.approx(1.0)

    def test_detected_cliches_sorted(self) -> None:
        assert self._result.detected_cliches == [
            "in conclusion",
            "in today's world",
            "it goes without saying",
            "needless to say",
            "the fact of the matter is",
        ]

    def test_detected_cliches_unique(self) -> None:
        d = self._result.detected_cliches
        assert len(d) == len(set(d))


class TestLongPhrase:
    """Six-token phrase 'it is important to note that' is detected correctly."""

    def setup_method(self) -> None:
        self._result = ClicheAnalyzer().analyze(_EN_LONG_PHRASE)

    def test_detected_cliches(self) -> None:
        assert self._result.detected_cliches == ["it is important to note that"]

    def test_cliche_count(self) -> None:
        assert self._result.cliche_count == 1

    def test_score(self) -> None:
        assert self._result.cliche_score == pytest.approx(0.2)


class TestApostropheVariants:
    """'in today's world' is detected under both tokenizer representations."""

    def test_nltk_variant_detected(self) -> None:
        # NLTK drops "'s" via isalpha filter → ("in","today","world")
        ctx = _en_ctx(("word",) * 20 + ("in", "today", "world") + ("word",) * 77)
        result = ClicheAnalyzer().analyze(ctx)
        assert "in today's world" in result.detected_cliches

    def test_regex_variant_detected(self) -> None:
        # regex splits on apostrophe → ("in","today","s","world")
        result = ClicheAnalyzer().analyze(_EN_APOSTROPHE_REGEX)
        assert "in today's world" in result.detected_cliches
        assert result.cliche_count == 1

    def test_both_variants_canonical_display_string(self) -> None:
        # Both variants display with the apostrophe in the canonical string.
        for ctx in (
            _en_ctx(("in", "today", "world")),
            _en_ctx(("in", "today", "s", "world")),
        ):
            result = ClicheAnalyzer().analyze(ctx)
            assert result.detected_cliches == ["in today's world"]

    def test_original_split_form_does_not_match(self) -> None:
        # ("in","today's","world") — no tokenizer produces a literal apostrophe
        # inside a word token, so this should NOT match.
        ctx = _en_ctx(("in", "today's", "world"))
        result = ClicheAnalyzer().analyze(ctx)
        assert result.cliche_count == 0


class TestTurkishTwoCliches:
    """100-token Turkish text with 'sonuç olarak' and 'günümüzde'."""

    def setup_method(self) -> None:
        self._result = ClicheAnalyzer().analyze(_TR_TWO_CLICHES)

    def test_detected_cliches(self) -> None:
        assert self._result.detected_cliches == ["günümüzde", "sonuç olarak"]

    def test_cliche_count(self) -> None:
        assert self._result.cliche_count == 2

    def test_density(self) -> None:
        assert self._result.cliche_density == pytest.approx(2.0)

    def test_score(self) -> None:
        assert self._result.cliche_score == pytest.approx(0.4)


class TestTurkishFourCliches:
    """100-token Turkish text with four distinct clichés."""

    def setup_method(self) -> None:
        self._result = ClicheAnalyzer().analyze(_TR_FOUR_CLICHES)

    def test_detected_cliches(self) -> None:
        assert self._result.detected_cliches == [
            "bilindiği üzere",
            "büyük önem taşımaktadır",
            "günümüzde",
            "sonuç olarak",
        ]

    def test_cliche_count(self) -> None:
        assert self._result.cliche_count == 4

    def test_density(self) -> None:
        assert self._result.cliche_density == pytest.approx(4.0)

    def test_score(self) -> None:
        assert self._result.cliche_score == pytest.approx(0.8)


class TestTurkishLongPhrases:
    """Multi-token Turkish clichés are detected at exact token boundaries."""

    def test_yadsınamaz_detected(self) -> None:
        ctx = _tr_ctx(
            ("kelime",) * 10
            + ("yadsınamaz", "bir", "gerçektir", "ki")
            + ("kelime",) * 6
        )
        result = ClicheAnalyzer().analyze(ctx)
        assert "yadsınamaz bir gerçektir ki" in result.detected_cliches

    def test_bunun_altını_detected(self) -> None:
        ctx = _tr_ctx(
            ("kelime",) * 5
            + ("bunun", "altını", "çizmek", "gerekir")
            + ("kelime",) * 3
        )
        result = ClicheAnalyzer().analyze(ctx)
        assert "bunun altını çizmek gerekir" in result.detected_cliches

    def test_büyük_önem_detected(self) -> None:
        ctx = _tr_ctx(("büyük", "önem", "taşımaktadır"))
        result = ClicheAnalyzer().analyze(ctx)
        assert "büyük önem taşımaktadır" in result.detected_cliches

    def test_bilindiği_üzere_detected(self) -> None:
        ctx = _tr_ctx(("bilindiği", "üzere"))
        result = ClicheAnalyzer().analyze(ctx)
        assert "bilindiği üzere" in result.detected_cliches

    def test_partial_match_not_detected(self) -> None:
        # Only "bilindiği" without "üzere" → no match
        ctx = _tr_ctx(("bilindiği", "olarak"))
        result = ClicheAnalyzer().analyze(ctx)
        assert result.cliche_count == 0


class TestScoreCapping:
    """Score is capped at 1.0 regardless of how high the density rises."""

    def test_density_above_cap_gives_score_one(self) -> None:
        # 10 tokens with 5 occurrences of "in conclusion" (2 tokens each)
        ctx = _en_ctx(("in", "conclusion") * 5)
        result = ClicheAnalyzer().analyze(ctx)
        assert result.cliche_score == pytest.approx(1.0)
        assert result.cliche_density == pytest.approx(50.0)

    def test_score_never_exceeds_one(self) -> None:
        ctx = _en_ctx(("in", "conclusion") * 20)
        result = ClicheAnalyzer().analyze(ctx)
        assert result.cliche_score <= 1.0


class TestDensityFormula:
    """cliche_density = cliche_count / n_tokens × 100."""

    def test_small_text_density(self) -> None:
        # 10 tokens, 1 cliché ("needless to say" = 3 tokens) → density = 10.0
        ctx = _en_ctx(("word",) * 7 + ("needless", "to", "say"))
        result = ClicheAnalyzer().analyze(ctx)
        assert result.cliche_density == pytest.approx(10.0)

    def test_large_text_low_density(self) -> None:
        # 1000 tokens, 1 cliché → density = 0.1
        ctx = _en_ctx(("in", "conclusion") + ("word",) * 998)
        result = ClicheAnalyzer().analyze(ctx)
        assert result.cliche_density == pytest.approx(0.1)
        assert result.cliche_score == pytest.approx(0.02)

    def test_density_consistent_with_count_and_tokens(self) -> None:
        result = ClicheAnalyzer().analyze(_EN_MULTI)
        expected = result.cliche_count / 100 * 100.0
        assert result.cliche_density == pytest.approx(expected)


class TestInvariants:
    """Parametrized invariants that must hold across all fixtures."""

    @pytest.mark.parametrize("ctx", _ALL_FIXTURES)
    def test_score_in_range(self, ctx: AnalysisContext) -> None:
        result = ClicheAnalyzer().analyze(ctx)
        assert 0.0 <= result.cliche_score <= 1.0

    @pytest.mark.parametrize("ctx", _ALL_FIXTURES)
    def test_density_non_negative(self, ctx: AnalysisContext) -> None:
        result = ClicheAnalyzer().analyze(ctx)
        assert result.cliche_density >= 0.0

    @pytest.mark.parametrize("ctx", _ALL_FIXTURES)
    def test_count_non_negative(self, ctx: AnalysisContext) -> None:
        result = ClicheAnalyzer().analyze(ctx)
        assert result.cliche_count >= 0

    @pytest.mark.parametrize("ctx", _ALL_FIXTURES)
    def test_detected_cliches_sorted(self, ctx: AnalysisContext) -> None:
        result = ClicheAnalyzer().analyze(ctx)
        assert result.detected_cliches == sorted(result.detected_cliches)

    @pytest.mark.parametrize("ctx", _ALL_FIXTURES)
    def test_detected_cliches_unique(self, ctx: AnalysisContext) -> None:
        result = ClicheAnalyzer().analyze(ctx)
        d = result.detected_cliches
        assert len(d) == len(set(d))

    @pytest.mark.parametrize("ctx", _ALL_FIXTURES)
    def test_count_gte_len_detected(self, ctx: AnalysisContext) -> None:
        # Total count ≥ number of unique phrases (each phrase ≥1 occurrence)
        result = ClicheAnalyzer().analyze(ctx)
        assert result.cliche_count >= len(result.detected_cliches)

    @pytest.mark.parametrize("ctx", _ALL_FIXTURES)
    def test_score_consistent_with_density(self, ctx: AnalysisContext) -> None:
        result = ClicheAnalyzer().analyze(ctx)
        expected = min(1.0, result.cliche_density / 5.0)
        assert result.cliche_score == pytest.approx(expected, abs=1e-9)

    @pytest.mark.parametrize("ctx", _ALL_FIXTURES)
    def test_deterministic(self, ctx: AnalysisContext) -> None:
        analyzer = ClicheAnalyzer()
        assert analyzer.analyze(ctx) == analyzer.analyze(ctx)

    @pytest.mark.parametrize("ctx", _ALL_FIXTURES)
    def test_context_not_mutated(self, ctx: AnalysisContext) -> None:
        original = ctx.tokens
        ClicheAnalyzer().analyze(ctx)
        assert ctx.tokens == original


class TestCountCliches:
    """Unit tests for _count_cliches helper."""

    def test_empty_tokens(self) -> None:
        counts = _count_cliches(())
        assert sum(counts.values()) == 0

    def test_single_match(self) -> None:
        counts = _count_cliches(("in", "conclusion"))
        assert counts["in conclusion"] == 1

    def test_repeated_match(self) -> None:
        tokens = ("in", "conclusion") * 3
        counts = _count_cliches(tokens)
        assert counts["in conclusion"] == 3

    def test_multiple_distinct_phrases(self) -> None:
        tokens = (
            ("needless", "to", "say")
            + ("word",) * 5
            + ("in", "conclusion")
        )
        counts = _count_cliches(tokens)
        assert counts["needless to say"] == 1
        assert counts["in conclusion"] == 1
        assert sum(counts.values()) == 2

    def test_no_partial_match(self) -> None:
        # Only first token of a phrase
        counts = _count_cliches(("in",))
        assert sum(counts.values()) == 0

    def test_turkish_single_token_cliche(self) -> None:
        counts = _count_cliches(("günümüzde",))
        assert counts["günümüzde"] == 1

    def test_turkish_two_token_cliche(self) -> None:
        counts = _count_cliches(("sonuç", "olarak"))
        assert counts["sonuç olarak"] == 1

    def test_nltk_apostrophe_variant(self) -> None:
        counts = _count_cliches(("in", "today", "world"))
        assert counts["in today's world"] == 1

    def test_regex_apostrophe_variant(self) -> None:
        counts = _count_cliches(("in", "today", "s", "world"))
        assert counts["in today's world"] == 1

    def test_overlapping_phrase_tokens_not_double_counted(self) -> None:
        # "the" appears twice in "the fact of the matter is" — should count
        # the phrase once, not detect "the" as matching sub-phrases.
        counts = _count_cliches(("the", "fact", "of", "the", "matter", "is"))
        assert counts["the fact of the matter is"] == 1
        assert sum(counts.values()) == 1


class TestRealisticEnglish:
    """Realistic academic English essay scenarios."""

    def test_formulaic_conclusion_paragraph(self) -> None:
        # Simulates a conclusion paragraph loaded with transition clichés.
        # "in conclusion", "needless to say", "it goes without saying" all present.
        tokens = (
            ("in", "conclusion")
            + ("word",) * 15
            + ("needless", "to", "say")
            + ("word",) * 15
            + ("it", "goes", "without", "saying")
            + ("word",) * 12
            + ("word",) * 15
        )
        ctx = _en_ctx(tokens)
        result = ClicheAnalyzer().analyze(ctx)
        assert result.cliche_count == 3
        assert set(result.detected_cliches) == {
            "in conclusion",
            "needless to say",
            "it goes without saying",
        }
        assert result.cliche_score > 0.5

    def test_clean_academic_writing_no_cliches(self) -> None:
        # Well-written academic prose does not trigger cliché detection.
        tokens = (
            "the", "study", "reveals", "a", "significant", "correlation",
            "between", "syntactic", "complexity", "and", "text", "difficulty",
            "as", "measured", "by", "validated", "readability", "instruments",
        )
        ctx = _en_ctx(tokens)
        result = ClicheAnalyzer().analyze(ctx)
        assert result.cliche_count == 0
        assert result.cliche_score == 0.0

    def test_important_to_note_phrase(self) -> None:
        tokens = (
            "word",) * 5 + ("it", "is", "important", "to", "note", "that") + ("word",) * 9
        ctx = _en_ctx(tokens)
        result = ClicheAnalyzer().analyze(ctx)
        assert "it is important to note that" in result.detected_cliches

    def test_todays_world_in_context(self) -> None:
        # Realistic English sentence using NLTK variant.
        tokens = (
            "students", "today", "in", "today", "world", "face",
            "unprecedented", "challenges", "requiring", "adaptable", "thinking",
        )
        ctx = _en_ctx(tokens)
        result = ClicheAnalyzer().analyze(ctx)
        assert "in today's world" in result.detected_cliches

    def test_high_cliche_density_signals_formulaic_writing(self) -> None:
        # A text where 10% of tokens form clichés should score high.
        cliche_block = ("in", "conclusion") * 5  # 10 tokens, 5 matches
        ctx = _en_ctx(cliche_block + ("word",) * 90)  # 100 tokens total
        result = ClicheAnalyzer().analyze(ctx)
        assert result.cliche_count == 5
        assert result.cliche_score == pytest.approx(1.0)


class TestRealisticTurkish:
    """Realistic academic Turkish essay scenarios."""

    def test_formulaic_turkish_essay_opening(self) -> None:
        # "günümüzde" and "bilindiği üzere" typical in formulaic Turkish essays.
        tokens = (
            ("günümüzde",)
            + ("kelime",) * 12
            + ("bilindiği", "üzere")
            + ("kelime",) * 10
        )
        ctx = _tr_ctx(tokens)
        result = ClicheAnalyzer().analyze(ctx)
        assert result.cliche_count == 2
        assert "günümüzde" in result.detected_cliches
        assert "bilindiği üzere" in result.detected_cliches

    def test_sonuç_olarak_in_conclusion_paragraph(self) -> None:
        tokens = (
            "sonuç", "olarak", "bu", "çalışma", "göstermektedir", "ki",
            "yöntem", "geçerlidir",
        )
        ctx = _tr_ctx(tokens)
        result = ClicheAnalyzer().analyze(ctx)
        assert "sonuç olarak" in result.detected_cliches

    def test_mixed_tr_cliches_score_scales_with_density(self) -> None:
        # Two clichés in a 50-token text → density=4.0, score=0.8
        tokens = (
            ("günümüzde",)
            + ("kelime",) * 24
            + ("sonuç", "olarak")
            + ("kelime",) * 23
        )
        ctx = _tr_ctx(tokens)
        result = ClicheAnalyzer().analyze(ctx)
        assert result.cliche_density == pytest.approx(4.0)
        assert result.cliche_score == pytest.approx(0.8)

    def test_clean_turkish_text_no_cliches(self) -> None:
        tokens = (
            "araştırma", "bulgular", "analiz", "veri", "yöntem",
            "örneklem", "değerlendirme", "istatistik",
        )
        ctx = _tr_ctx(tokens)
        result = ClicheAnalyzer().analyze(ctx)
        assert result.cliche_count == 0

    def test_all_six_turkish_cliches_detectable(self) -> None:
        tokens = (
            ("sonuç", "olarak")
            + ("günümüzde",)
            + ("bilindiği", "üzere")
            + ("büyük", "önem", "taşımaktadır")
            + ("yadsınamaz", "bir", "gerçektir", "ki")
            + ("bunun", "altını", "çizmek", "gerekir")
        )
        ctx = _tr_ctx(tokens)
        result = ClicheAnalyzer().analyze(ctx)
        assert result.cliche_count == 6
        assert set(result.detected_cliches) == set(_TR_CLICHES)
