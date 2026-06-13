"""Readability analyzer for English (Flesch) and Turkish (simplified index).

Turkish Readability Index (TRI) — formula and limitations
==========================================================

No widely validated Turkish readability formula exists that matches the
psychometric rigour of the English Flesch scales.  The Ateşman (1997) formula
(RS = 198.825 − 40.175 × SPW − 2.610 × ASL) was calibrated on a small,
dated corpus and is not officially standardised.

This module implements an independent *Turkish Readability Index* (TRI):

    TRI = 100 − 2.6 × ASL − 12.0 × max(0, SPW − 1.5)

where:
    ASL  = average sentence length in words (total tokens / total sentences)
    SPW  = average syllables per word, approximated by counting Turkish
           vowel letters (a, e, ı, i, o, ö, u, ü) per token.

Rationale for design choices:
    - Turkish orthography is near-phonemic: every vowel letter is a separate
      syllable nucleus, so vowel-counting is a reliable syllable proxy that
      does not require a pronunciation dictionary.
    - The −1.5 baseline subtracts the natural minimum vowel load.  Simple
      Turkish words already carry ~1.5 vowels on average; only excess beyond
      that baseline is penalised.
    - ASL has a larger absolute coefficient because sentence length is the
      strongest predictor of perceived text difficulty across languages.
    - SPW is de-weighted relative to English Flesch (coefficient 12 vs 84.6)
      because Turkish agglutination routinely produces high-syllable-count
      words (e.g. "kullanılmaktadır" = 7 syllables) that are morphologically
      common and not necessarily harder for native readers.

Known limitations:
    1. Not validated against Turkish reading-comprehension studies.
    2. Morphologically complex but frequent words are over-penalised.
    3. The score may underestimate difficulty for texts with rare vocabulary
       or domain-specific terminology, since TRI is a surface-form metric.
    4. Turkish academic registers often use very long sentences with heavy
       subordination; TRI may underestimate difficulty when the clause
       structure is complex but word lengths are moderate.
    5. Results should be treated as a heuristic signal, not a standardised
       psychometric measure.
"""

import textstat

from src.analyzers.base import BaseAnalyzer
from src.models.analysis import AnalysisContext
from src.models.enums import Language
from src.models.response import ReadabilityResult

# Turkish vowel set — all eight vowels in the Turkish alphabet.
_TR_VOWELS: frozenset[str] = frozenset("aeıioöuü")


class ReadabilityAnalyzer(BaseAnalyzer[ReadabilityResult]):
    """Estimates text readability using language-appropriate formulas.

    English dispatch:
        * Flesch Reading Ease (FRE): 206.835 − 1.015×ASL − 84.6×SPW
          Higher = easier (scale 0–100, clamped).
        * Flesch-Kincaid Grade Level: 0.39×ASL + 11.8×SPW − 15.59
          Converted to "Grade N" or "College+" label.

    Turkish dispatch:
        * Turkish Readability Index (TRI) — see module-level docstring for
          the formula, rationale, and limitations.

    Both scores are normalised to [0, 100] and mapped to a shared five-level
    classification: very_difficult / difficult / standard / easy / very_easy.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this analyzer."""
        return "readability"

    def analyze(self, context: AnalysisContext) -> ReadabilityResult:
        """Compute readability metrics for the given context.

        Returns a default very_difficult result for empty texts rather than
        raising an error, so the pipeline can continue gracefully.

        Args:
            context: Immutable pipeline context from TokenizerService.

        Returns:
            ReadabilityResult with readability_score, grade_level, and
            classification.
        """
        if not context.tokens:
            return ReadabilityResult(
                readability_score=0.0,
                grade_level="N/A",
                classification="very_difficult",
            )
        if context.language == Language.TURKISH:
            return _analyze_turkish(context)
        return _analyze_english(context)


# ---------------------------------------------------------------------------
# Language-specific dispatch functions — independently testable, no state.
# ---------------------------------------------------------------------------


def _analyze_english(context: AnalysisContext) -> ReadabilityResult:
    """Apply Flesch Reading Ease and Flesch-Kincaid Grade to English text.

    Uses the sentence strings from the context (rather than raw_text) so
    the readability computation operates on the same segmentation already
    used by all other analyzers in the pipeline.

    Args:
        context: Non-empty pipeline context with Language.ENGLISH.

    Returns:
        ReadabilityResult with FRE score, FK grade label, and classification.
    """
    text = " ".join(context.sentences)
    fre = max(0.0, min(100.0, textstat.flesch_reading_ease(text)))
    fk_grade = textstat.flesch_kincaid_grade(text)
    grade_num = max(1, round(fk_grade))
    grade_level = f"Grade {grade_num}" if grade_num <= 12 else "College+"
    return ReadabilityResult(
        readability_score=round(fre, 2),
        grade_level=grade_level,
        classification=_classify_score(fre),
    )


def _analyze_turkish(context: AnalysisContext) -> ReadabilityResult:
    """Apply the Turkish Readability Index to Turkish text.

    Args:
        context: Non-empty pipeline context with Language.TURKISH.

    Returns:
        ReadabilityResult with TRI score, grade stage label, and
        classification.
    """
    tri = _compute_tri(context.tokens, context.sentence_token_counts)
    return ReadabilityResult(
        readability_score=round(tri, 2),
        grade_level=_tr_grade_level(tri),
        classification=_classify_score(tri),
    )


def _compute_tri(
    tokens: tuple[str, ...],
    sentence_token_counts: tuple[int, ...],
) -> float:
    """Compute the Turkish Readability Index (TRI).

    TRI = 100 − 2.6 × ASL − 12.0 × max(0, SPW − 1.5)

    Syllables are approximated by counting Turkish vowel characters
    (a, e, ı, i, o, ö, u, ü) per token.  Each token is credited with at
    least one syllable to handle abbreviations and numerals that may have
    slipped through tokenisation.

    Args:
        tokens: Lowercased word tokens from the context.
        sentence_token_counts: Per-sentence token counts.

    Returns:
        TRI score clamped to [0.0, 100.0].  Higher = easier.
    """
    n_sentences = len(sentence_token_counts)
    n_tokens = len(tokens)
    if n_sentences == 0 or n_tokens == 0:
        return 0.0

    asl = n_tokens / n_sentences
    total_syllables = sum(
        max(1, sum(1 for ch in token if ch in _TR_VOWELS)) for token in tokens
    )
    spw = total_syllables / n_tokens
    tri = 100.0 - 2.6 * asl - 12.0 * max(0.0, spw - 1.5)
    return max(0.0, min(100.0, tri))


def _tr_grade_level(tri: float) -> str:
    """Map a TRI score to an approximate Turkish educational stage.

    Labels follow the Turkish national education system (MEB).  Grade ranges
    are broad approximations since TRI is not validated against grade norms.

    Args:
        tri: TRI score in [0, 100].

    Returns:
        Descriptive stage label in Turkish with English translation.
    """
    if tri >= 80.0:
        return "İlkokul (Primary, Grade 1-4)"
    if tri >= 60.0:
        return "Ortaokul (Middle School, Grade 5-8)"
    if tri >= 40.0:
        return "Lise (High School, Grade 9-12)"
    if tri >= 20.0:
        return "Üniversite (University)"
    return "İleri Akademik (Advanced Academic)"


def _classify_score(score: float) -> str:
    """Map a readability score to a difficulty label.

    Thresholds apply to both Flesch Reading Ease (English) and TRI (Turkish)
    since both are normalised to [0, 100].

    Args:
        score: Readability score in [0, 100].

    Returns:
        One of: very_difficult, difficult, standard, easy, very_easy.
    """
    if score < 30.0:
        return "very_difficult"
    if score < 50.0:
        return "difficult"
    if score < 70.0:
        return "standard"
    if score < 90.0:
        return "easy"
    return "very_easy"
