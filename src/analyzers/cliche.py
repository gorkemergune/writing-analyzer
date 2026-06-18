"""Cliché detection analyzer for English and Turkish academic text."""

from collections import Counter

from src.analyzers.base import BaseAnalyzer
from src.models.analysis import AnalysisContext
from src.models.response import ClicheResult

_EN_CLICHES: tuple[str, ...] = (
    "in conclusion",
    "it is important to note that",
    "in today's world",
    "needless to say",
    "the fact of the matter is",
    "it goes without saying",
)

_TR_CLICHES: tuple[str, ...] = (
    "sonuç olarak",
    "günümüzde",
    "bilindiği üzere",
    "büyük önem taşımaktadır",
    "yadsınamaz bir gerçektir ki",
    "bunun altını çizmek gerekir",
)

_ALL_CLICHES: tuple[str, ...] = _EN_CLICHES + _TR_CLICHES


def _build_phrase_map() -> dict[tuple[str, ...], str]:
    """Build a mapping from token-tuples to canonical cliché strings.

    Phrases without apostrophes are registered directly from their
    space-split form.  Phrases containing apostrophes are registered under
    each tokenizer-specific token sequence.

    Returns:
        Dict mapping each token-tuple key to its display phrase string.
    """
    mapping: dict[tuple[str, ...], str] = {}
    for phrase in _ALL_CLICHES:
        if "'" not in phrase:
            mapping[tuple(phrase.split())] = phrase
    # NLTK and regex tokenizers split "today's" differently; both variants map to the same phrase.
    mapping[("in", "today", "world")] = "in today's world"
    mapping[("in", "today", "s", "world")] = "in today's world"
    return mapping


_PHRASE_MAP: dict[tuple[str, ...], str] = _build_phrase_map()

_SCORE_DENSITY_CAP: float = 5.0


class ClicheAnalyzer(BaseAnalyzer[ClicheResult]):
    """Detects formulaic cliché expressions in English and Turkish text.

    Scans the flat token sequence for every registered cliché phrase using
    a sliding-window comparison over _PHRASE_MAP.  The same phrase may
    appear multiple times; each occurrence increments the count.

    cliche_score combines density into a [0, 1] signal:
        cliche_score = min(1.0, cliche_density / 5.0)

    where cliche_density = total occurrences × 100 / total tokens.
    The score saturates at 5 clichés per 100 words.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this analyzer."""
        return "cliche"

    def analyze(self, context: AnalysisContext) -> ClicheResult:
        """Detect clichés in the context.

        Args:
            context: Immutable pipeline context from TokenizerService.

        Returns:
            ClicheResult with detected_cliches (unique, sorted), cliche_count,
            cliche_density (per 100 words), and cliche_score in [0, 1].
        """
        tokens = context.tokens
        if not tokens:
            return ClicheResult(
                detected_cliches=[],
                cliche_count=0,
                cliche_density=0.0,
                cliche_score=0.0,
            )

        counts = _count_cliches(tokens)
        cliche_count = sum(counts.values())
        detected_cliches = sorted(counts)

        density = cliche_count / len(tokens) * 100.0
        score = min(1.0, density / _SCORE_DENSITY_CAP)

        return ClicheResult(
            detected_cliches=detected_cliches,
            cliche_count=cliche_count,
            cliche_density=density,
            cliche_score=score,
        )



def _count_cliches(tokens: tuple[str, ...]) -> Counter[str]:
    """Return a counter mapping each matched cliché phrase to its count.

    Scans every registered phrase against the full token sequence using a
    sliding window.  A phrase may match at multiple positions; all matches
    are counted.  Both NLTK and regex tokenizer variants of "in today's
    world" resolve to the same canonical key.

    Args:
        tokens: Lowercased surface tokens from the context.

    Returns:
        Counter of phrase string → occurrence count for all matched clichés.
    """
    counts: Counter[str] = Counter()
    n = len(tokens)
    for phrase_tokens, phrase_str in _PHRASE_MAP.items():
        phrase_len = len(phrase_tokens)
        for i in range(n - phrase_len + 1):
            if tokens[i : i + phrase_len] == phrase_tokens:
                counts[phrase_str] += 1
    return counts
