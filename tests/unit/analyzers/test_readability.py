"""Unit tests for ReadabilityAnalyzer."""

import pytest

from src.analyzers.readability import (
    ReadabilityAnalyzer,
    _classify_score,
    _compute_tri,
    _tr_grade_level,
)
from src.models.analysis import AnalysisContext
from src.models.enums import DocumentType, Language
from src.models.response import ReadabilityResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_CLASSIFICATIONS = frozenset(
    {"very_difficult", "difficult", "standard", "easy", "very_easy"}
)


def _make_en(sentences: tuple[str, ...]) -> AnalysisContext:
    """Build a minimal English AnalysisContext from sentence strings."""
    tokens = tuple(
        word.lower().strip(".,!?;:\"'")
        for sent in sentences
        for word in sent.split()
        if word.strip(".,!?;:\"'")
    )
    counts = tuple(len(sent.split()) for sent in sentences)
    return AnalysisContext(
        raw_text=" ".join(sentences),
        language=Language.ENGLISH,
        document_type=DocumentType.ESSAY,
        cleaned_text=" ".join(sentences),
        tokens=tokens,
        sentences=sentences,
        stems=tokens,
        sentence_token_counts=counts,
    )


def _make_tr(
    tokens: tuple[str, ...],
    sentence_token_counts: tuple[int, ...],
) -> AnalysisContext:
    """Build a minimal Turkish AnalysisContext from token list."""
    return AnalysisContext(
        raw_text="stub",
        language=Language.TURKISH,
        document_type=DocumentType.ESSAY,
        cleaned_text="stub",
        tokens=tokens,
        sentences=("stub",) * len(sentence_token_counts),
        stems=tokens,
        sentence_token_counts=sentence_token_counts,
    )


# ---------------------------------------------------------------------------
# Module-level AnalysisContext fixtures
# ---------------------------------------------------------------------------

# ── English — empty ──────────────────────────────────────────────────────────
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

# ── English — simple prose ────────────────────────────────────────────────────
# Short common words, short sentences → FRE well above 70.
_EN_SIMPLE = _make_en(
    (
        "The cat sat on the mat.",
        "Dogs run and play all day.",
        "Birds can fly in the sky.",
        "Fish swim fast in the sea.",
        "The sun shines bright today.",
    )
)

# ── English — dense academic text ─────────────────────────────────────────────
# Long multisyllabic words, long sentences → FRE well below 40.
_EN_ACADEMIC = _make_en(
    (
        "The epistemological implications of contemporary neuroscientific "
        "discourse necessitate a fundamental reconsideration of established "
        "philosophical frameworks regarding consciousness.",
        "Longitudinal investigations into the phenomenological dimensions of "
        "subjective experience reveal increasingly complex interdependencies "
        "between neurobiological substrates and cognitive processes.",
        "Methodological pluralism in contemporary social science research "
        "acknowledges the inherent limitations of positivist epistemology "
        "in capturing multidimensional aspects of human experience.",
    )
)

# ── Turkish — easy (SPW ≤ 1.5, short sentences) ───────────────────────────────
# tokens = ("ev","güzel","hava","iyi","her","gün"), 2 sentences of 3
# syllables: e=1, ü+e=2, a+a=2, i+i=2, e=1, ü=1  → total=9
# ASL=3.0, SPW=9/6=1.5, penalty=max(0,0.0)=0
# TRI = 100 − 7.8 − 0 = 92.2  →  very_easy
_TR_SIMPLE = _make_tr(
    tokens=("ev", "güzel", "hava", "iyi", "her", "gün"),
    sentence_token_counts=(3, 3),
)

# ── Turkish — academic (high SPW, moderate ASL) ───────────────────────────────
# 12 complex tokens in 2 sentences of 6 each.
# Vowel counts: araştırma=4, bulguları=4, göstermektedir=5, sonuçların=4,
#               değerlendirilmesi=7, gereklidir=4,
#               kullanılmaktadır=6, yöntemin=3, geçerliliği=5,
#               sorgulanmalı=5, olgusal=3, kanıtlarla=4
# total syllables = 54, SPW=54/12=4.5, ASL=6.0
# TRI = 100 − 15.6 − 12×(4.5−1.5) = 100 − 15.6 − 36 = 48.4  →  difficult
_TR_ACADEMIC = _make_tr(
    tokens=(
        "araştırma",
        "bulguları",
        "göstermektedir",
        "sonuçların",
        "değerlendirilmesi",
        "gereklidir",
        "kullanılmaktadır",
        "yöntemin",
        "geçerliliği",
        "sorgulanmalı",
        "olgusal",
        "kanıtlarla",
    ),
    sentence_token_counts=(6, 6),
)

# ── Turkish — synthetic easy ("a" ×20, 4 sentences of 5) ─────────────────────
# SPW=1.0 (≤ 1.5, no penalty), ASL=5.0
# TRI = 100 − 13.0 − 0 = 87.0  →  easy
_TR_SYNTH_EASY = _make_tr(tokens=("a",) * 20, sentence_token_counts=(5, 5, 5, 5))

# ── Turkish — synthetic difficult ("aaa" ×30, 2 sentences of 15) ─────────────
# SPW=3.0, penalty=max(0,1.5)=1.5, ASL=15.0
# TRI = 100 − 39.0 − 18.0 = 43.0  →  difficult
_TR_SYNTH_DIFFICULT = _make_tr(
    tokens=("aaa",) * 30, sentence_token_counts=(15, 15)
)

# ── Turkish — synthetic very_difficult ("aaaa" ×40, 1 sentence) ──────────────
# SPW=4.0, penalty=2.5, ASL=40.0
# TRI = 100 − 104.0 − 30.0 = −34 → clamped to 0.0  →  very_difficult
_TR_SYNTH_VERY_DIFFICULT = _make_tr(
    tokens=("aaaa",) * 40, sentence_token_counts=(40,)
)

_ALL_FIXTURES = [
    _EMPTY,
    _EN_SIMPLE,
    _EN_ACADEMIC,
    _TR_SIMPLE,
    _TR_ACADEMIC,
    _TR_SYNTH_EASY,
    _TR_SYNTH_DIFFICULT,
    _TR_SYNTH_VERY_DIFFICULT,
]


# ===========================================================================
# Test classes
# ===========================================================================


class TestAnalyzerIdentity:
    """Static properties of the analyzer object."""

    def test_name(self) -> None:
        assert ReadabilityAnalyzer().name == "readability"

    def test_analyze_returns_readability_result(self) -> None:
        result = ReadabilityAnalyzer().analyze(_EN_SIMPLE)
        assert isinstance(result, ReadabilityResult)

    def test_two_instances_produce_same_result(self) -> None:
        a, b = ReadabilityAnalyzer(), ReadabilityAnalyzer()
        assert a.analyze(_TR_SIMPLE) == b.analyze(_TR_SIMPLE)


class TestEmptyInput:
    """Empty context returns a neutral/zero default."""

    def setup_method(self) -> None:
        self._result = ReadabilityAnalyzer().analyze(_EMPTY)

    def test_readability_score_is_zero(self) -> None:
        assert self._result.readability_score == 0.0

    def test_grade_level_is_na(self) -> None:
        assert self._result.grade_level == "N/A"

    def test_classification_is_very_difficult(self) -> None:
        assert self._result.classification == "very_difficult"


class TestEnglishSimpleText:
    """Short common words, short sentences → easy range."""

    def setup_method(self) -> None:
        self._result = ReadabilityAnalyzer().analyze(_EN_SIMPLE)

    def test_score_above_70(self) -> None:
        assert self._result.readability_score > 70.0

    def test_classification_is_easy_or_better(self) -> None:
        assert self._result.classification in {"easy", "very_easy"}

    def test_grade_level_is_primary(self) -> None:
        # Simple children's text should score at primary grade levels.
        assert self._result.grade_level.startswith("Grade ")
        grade_num = int(self._result.grade_level.split()[1])
        assert 1 <= grade_num <= 8

    def test_score_in_valid_range(self) -> None:
        assert 0.0 <= self._result.readability_score <= 100.0


class TestEnglishAcademicText:
    """Long multisyllabic words, long sentences → difficult/very_difficult."""

    def setup_method(self) -> None:
        self._result = ReadabilityAnalyzer().analyze(_EN_ACADEMIC)

    def test_score_below_40(self) -> None:
        assert self._result.readability_score < 40.0

    def test_classification_is_difficult_or_worse(self) -> None:
        assert self._result.classification in {"difficult", "very_difficult"}

    def test_grade_level_is_college(self) -> None:
        assert self._result.grade_level == "College+"

    def test_score_lower_than_simple_text(self) -> None:
        simple_score = ReadabilityAnalyzer().analyze(_EN_SIMPLE).readability_score
        assert self._result.readability_score < simple_score


class TestEnglishGradeLevel:
    """Grade level formatting for English."""

    def test_grade_level_format_simple(self) -> None:
        result = ReadabilityAnalyzer().analyze(_EN_SIMPLE)
        assert result.grade_level.startswith("Grade ") or result.grade_level == "College+"

    def test_grade_level_format_academic(self) -> None:
        result = ReadabilityAnalyzer().analyze(_EN_ACADEMIC)
        assert result.grade_level == "College+"

    def test_single_easy_sentence(self) -> None:
        ctx = _make_en(("Go. Run. Play. Sit. Jump. Stop.",))
        result = ReadabilityAnalyzer().analyze(ctx)
        assert result.grade_level.startswith("Grade ")

    def test_grade_level_numeric_for_simple_text(self) -> None:
        result = ReadabilityAnalyzer().analyze(_EN_SIMPLE)
        if result.grade_level != "College+":
            grade_str = result.grade_level.replace("Grade ", "")
            assert grade_str.isdigit()


class TestTurkishSimpleText:
    """Short, common Turkish words → very_easy range."""

    def setup_method(self) -> None:
        self._result = ReadabilityAnalyzer().analyze(_TR_SIMPLE)

    def test_readability_score(self) -> None:
        assert self._result.readability_score == pytest.approx(92.2)

    def test_classification_is_very_easy(self) -> None:
        assert self._result.classification == "very_easy"

    def test_grade_level(self) -> None:
        assert self._result.grade_level == "İlkokul (Primary, Grade 1-4)"


class TestTurkishAcademicText:
    """Complex academic Turkish vocabulary → difficult range."""

    def setup_method(self) -> None:
        self._result = ReadabilityAnalyzer().analyze(_TR_ACADEMIC)

    def test_readability_score(self) -> None:
        assert self._result.readability_score == pytest.approx(48.4)

    def test_classification_is_difficult(self) -> None:
        assert self._result.classification == "difficult"

    def test_grade_level(self) -> None:
        assert self._result.grade_level == "Lise (High School, Grade 9-12)"

    def test_score_lower_than_simple(self) -> None:
        simple_score = ReadabilityAnalyzer().analyze(_TR_SIMPLE).readability_score
        assert self._result.readability_score < simple_score


class TestTurkishSyntheticFixtures:
    """Verify TRI formula using synthetic tokens with exact arithmetic."""

    def test_synth_easy_score(self) -> None:
        # "a"×20, 4 sentences: ASL=5, SPW=1.0, penalty=0 → TRI=87.0
        result = ReadabilityAnalyzer().analyze(_TR_SYNTH_EASY)
        assert result.readability_score == pytest.approx(87.0)

    def test_synth_easy_classification(self) -> None:
        result = ReadabilityAnalyzer().analyze(_TR_SYNTH_EASY)
        assert result.classification == "easy"

    def test_synth_difficult_score(self) -> None:
        # "aaa"×30, 2 sentences: ASL=15, SPW=3.0, penalty=1.5 → TRI=43.0
        result = ReadabilityAnalyzer().analyze(_TR_SYNTH_DIFFICULT)
        assert result.readability_score == pytest.approx(43.0)

    def test_synth_difficult_classification(self) -> None:
        result = ReadabilityAnalyzer().analyze(_TR_SYNTH_DIFFICULT)
        assert result.classification == "difficult"

    def test_synth_very_difficult_score_clamped_to_zero(self) -> None:
        # "aaaa"×40, 1 sentence: raw TRI = -34 → clamped to 0.0
        result = ReadabilityAnalyzer().analyze(_TR_SYNTH_VERY_DIFFICULT)
        assert result.readability_score == pytest.approx(0.0)

    def test_synth_very_difficult_classification(self) -> None:
        result = ReadabilityAnalyzer().analyze(_TR_SYNTH_VERY_DIFFICULT)
        assert result.classification == "very_difficult"


class TestComputeTri:
    """Unit tests for _compute_tri helper."""

    def test_empty_tokens_returns_zero(self) -> None:
        assert _compute_tri((), ()) == 0.0

    def test_zero_sentences_returns_zero(self) -> None:
        assert _compute_tri(("kelime",), ()) == 0.0

    def test_all_single_vowel_no_penalty(self) -> None:
        # SPW = 1.0 < 1.5 → no penalty; ASL = 5
        result = _compute_tri(("a",) * 20, (5, 5, 5, 5))
        assert result == pytest.approx(87.0)

    def test_three_vowel_tokens_exact(self) -> None:
        # "aaa"×30, 2 sentences: TRI = 43.0
        result = _compute_tri(("aaa",) * 30, (15, 15))
        assert result == pytest.approx(43.0)

    def test_four_vowel_tokens_clamped(self) -> None:
        # Raw TRI < 0 → clamped to 0.0
        result = _compute_tri(("aaaa",) * 40, (40,))
        assert result == pytest.approx(0.0)

    def test_clamped_at_100(self) -> None:
        # One-token, one-sentence, no vowels: ASL=1, SPW=1 (min 1), penalty=0
        # TRI = 100 - 2.6 - 0 = 97.4 < 100 — try two-char vowel token:
        # Actually test with a single zero-length-like token set where TRI = 100
        # ASL=0 requires 0 tokens → handled by empty check.
        # Instead verify score never exceeds 100 for any input.
        result = _compute_tri(("a",), (1,))
        assert result <= 100.0

    def test_token_with_no_vowels_counts_as_one(self) -> None:
        # "b" has no Turkish vowels → min(1,...) = 1 syllable
        result_b = _compute_tri(("b",) * 10, (5, 5))
        result_a = _compute_tri(("a",) * 10, (5, 5))
        # Both have SPW=1.0, ASL=2 → same TRI
        assert result_b == pytest.approx(result_a)

    def test_asl_increases_difficulty(self) -> None:
        # All else equal, more words per sentence → lower TRI
        short = _compute_tri(("a",) * 10, (2, 2, 2, 2, 2))  # ASL=2
        long_ = _compute_tri(("a",) * 10, (10,))  # ASL=10
        assert short > long_

    def test_spw_above_baseline_increases_difficulty(self) -> None:
        # Same ASL, higher syllables → lower TRI
        simple = _compute_tri(("a",) * 10, (5, 5))  # SPW=1.0
        complex_ = _compute_tri(("aaaa",) * 10, (5, 5))  # SPW=4.0
        assert simple > complex_

    def test_spw_at_or_below_baseline_no_penalty(self) -> None:
        # SPW exactly 1.5 → penalty = 0
        # "aaa" = 3 vowels; need avg=1.5 → mix "a" and "aa"
        # "a"×10 + "aa"×10 = 10 tokens, syllables=10+20=30, SPW=30/20=1.5
        result = _compute_tri(("a",) * 10 + ("aa",) * 10, (10, 10))
        asl = 20 / 2
        expected = 100.0 - 2.6 * asl
        assert result == pytest.approx(expected)

    def test_known_simple_turkish_words(self) -> None:
        # ev=1, güzel=2, hava=2, iyi=2, her=1, gün=1 → syl=9, SPW=1.5
        result = _compute_tri(
            ("ev", "güzel", "hava", "iyi", "her", "gün"), (3, 3)
        )
        assert result == pytest.approx(92.2)

    def test_known_academic_turkish_words(self) -> None:
        # 12 tokens, 2 sentences of 6, total syllables=54, SPW=4.5
        result = _compute_tri(
            (
                "araştırma",
                "bulguları",
                "göstermektedir",
                "sonuçların",
                "değerlendirilmesi",
                "gereklidir",
                "kullanılmaktadır",
                "yöntemin",
                "geçerliliği",
                "sorgulanmalı",
                "olgusal",
                "kanıtlarla",
            ),
            (6, 6),
        )
        assert result == pytest.approx(48.4)


class TestTrGradeLevel:
    """Unit tests for _tr_grade_level helper."""

    def test_primary_at_80(self) -> None:
        assert _tr_grade_level(80.0) == "İlkokul (Primary, Grade 1-4)"

    def test_primary_above_80(self) -> None:
        assert _tr_grade_level(95.0) == "İlkokul (Primary, Grade 1-4)"

    def test_primary_exactly_100(self) -> None:
        assert _tr_grade_level(100.0) == "İlkokul (Primary, Grade 1-4)"

    def test_middle_at_60(self) -> None:
        assert _tr_grade_level(60.0) == "Ortaokul (Middle School, Grade 5-8)"

    def test_middle_just_below_80(self) -> None:
        assert _tr_grade_level(79.9) == "Ortaokul (Middle School, Grade 5-8)"

    def test_high_school_at_40(self) -> None:
        assert _tr_grade_level(40.0) == "Lise (High School, Grade 9-12)"

    def test_high_school_just_below_60(self) -> None:
        assert _tr_grade_level(59.9) == "Lise (High School, Grade 9-12)"

    def test_university_at_20(self) -> None:
        assert _tr_grade_level(20.0) == "Üniversite (University)"

    def test_university_just_below_40(self) -> None:
        assert _tr_grade_level(39.9) == "Üniversite (University)"

    def test_advanced_below_20(self) -> None:
        assert _tr_grade_level(10.0) == "İleri Akademik (Advanced Academic)"

    def test_advanced_at_zero(self) -> None:
        assert _tr_grade_level(0.0) == "İleri Akademik (Advanced Academic)"


class TestClassifyScore:
    """Unit tests for _classify_score helper."""

    def test_below_30_is_very_difficult(self) -> None:
        assert _classify_score(0.0) == "very_difficult"
        assert _classify_score(29.9) == "very_difficult"

    def test_30_is_difficult(self) -> None:
        assert _classify_score(30.0) == "difficult"

    def test_just_below_50_is_difficult(self) -> None:
        assert _classify_score(49.9) == "difficult"

    def test_50_is_standard(self) -> None:
        assert _classify_score(50.0) == "standard"

    def test_just_below_70_is_standard(self) -> None:
        assert _classify_score(69.9) == "standard"

    def test_70_is_easy(self) -> None:
        assert _classify_score(70.0) == "easy"

    def test_just_below_90_is_easy(self) -> None:
        assert _classify_score(89.9) == "easy"

    def test_90_is_very_easy(self) -> None:
        assert _classify_score(90.0) == "very_easy"

    def test_100_is_very_easy(self) -> None:
        assert _classify_score(100.0) == "very_easy"

    def test_exact_boundaries(self) -> None:
        for threshold, expected in [
            (30.0, "difficult"),
            (50.0, "standard"),
            (70.0, "easy"),
            (90.0, "very_easy"),
        ]:
            assert _classify_score(threshold) == expected


class TestLanguageRouting:
    """Correct formula is selected based on context.language."""

    def test_english_context_uses_fre_scale(self) -> None:
        # English text: FRE can produce values not achievable by TRI for same
        # surface structure.  Key invariant: English → FK grade label format.
        result = ReadabilityAnalyzer().analyze(_EN_SIMPLE)
        grade = result.grade_level
        assert grade.startswith("Grade ") or grade == "College+"

    def test_turkish_context_uses_tri_scale(self) -> None:
        # Turkish → Turkish stage label format.
        result = ReadabilityAnalyzer().analyze(_TR_SIMPLE)
        stage_labels = {
            "İlkokul (Primary, Grade 1-4)",
            "Ortaokul (Middle School, Grade 5-8)",
            "Lise (High School, Grade 9-12)",
            "Üniversite (University)",
            "İleri Akademik (Advanced Academic)",
        }
        assert result.grade_level in stage_labels

    def test_same_tokens_different_language_different_result(self) -> None:
        # A context with real English words processed as Turkish vs English
        # should give different readability scores because they use different
        # formulas.
        en_ctx = _make_en(("The research findings demonstrate significant implications.",))
        tr_ctx = _make_tr(
            tokens=("the", "research", "findings", "demonstrate", "significant", "implications"),
            sentence_token_counts=(6,),
        )
        en_result = ReadabilityAnalyzer().analyze(en_ctx)
        tr_result = ReadabilityAnalyzer().analyze(tr_ctx)
        assert en_result.readability_score != tr_result.readability_score


class TestScoreClamping:
    """Scores are always within [0, 100] regardless of input extremes."""

    def test_very_complex_turkish_clamped_at_zero(self) -> None:
        result = ReadabilityAnalyzer().analyze(_TR_SYNTH_VERY_DIFFICULT)
        assert result.readability_score >= 0.0

    def test_very_simple_turkish_does_not_exceed_100(self) -> None:
        # Single one-token, one-sentence context → TRI = 100 - 2.6 - 0 = 97.4
        ctx = _make_tr(tokens=("a",), sentence_token_counts=(1,))
        result = ReadabilityAnalyzer().analyze(ctx)
        assert result.readability_score <= 100.0

    def test_english_score_does_not_exceed_100(self) -> None:
        result = ReadabilityAnalyzer().analyze(_EN_SIMPLE)
        assert result.readability_score <= 100.0

    def test_english_score_not_negative(self) -> None:
        result = ReadabilityAnalyzer().analyze(_EN_ACADEMIC)
        assert result.readability_score >= 0.0


class TestInvariants:
    """Parametrized invariants that must hold for every fixture."""

    @pytest.mark.parametrize("ctx", _ALL_FIXTURES)
    def test_score_in_range(self, ctx: AnalysisContext) -> None:
        result = ReadabilityAnalyzer().analyze(ctx)
        assert 0.0 <= result.readability_score <= 100.0

    @pytest.mark.parametrize("ctx", _ALL_FIXTURES)
    def test_classification_is_valid(self, ctx: AnalysisContext) -> None:
        result = ReadabilityAnalyzer().analyze(ctx)
        assert result.classification in _VALID_CLASSIFICATIONS

    @pytest.mark.parametrize("ctx", _ALL_FIXTURES)
    def test_classification_matches_score(self, ctx: AnalysisContext) -> None:
        result = ReadabilityAnalyzer().analyze(ctx)
        expected = _classify_score(result.readability_score)
        assert result.classification == expected

    @pytest.mark.parametrize("ctx", _ALL_FIXTURES)
    def test_grade_level_is_non_empty_string(self, ctx: AnalysisContext) -> None:
        result = ReadabilityAnalyzer().analyze(ctx)
        assert isinstance(result.grade_level, str)
        assert len(result.grade_level) > 0

    @pytest.mark.parametrize("ctx", _ALL_FIXTURES)
    def test_deterministic(self, ctx: AnalysisContext) -> None:
        analyzer = ReadabilityAnalyzer()
        assert analyzer.analyze(ctx) == analyzer.analyze(ctx)

    @pytest.mark.parametrize("ctx", _ALL_FIXTURES)
    def test_context_not_mutated(self, ctx: AnalysisContext) -> None:
        original_tokens = ctx.tokens
        ReadabilityAnalyzer().analyze(ctx)
        assert ctx.tokens == original_tokens
