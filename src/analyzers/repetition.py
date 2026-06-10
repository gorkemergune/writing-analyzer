"""Repetition detection analyzer."""

from collections import Counter

from src.analyzers.base import BaseAnalyzer
from src.models.analysis import AnalysisContext
from src.models.response import RepeatedItem, RepetitionResult

# ---------------------------------------------------------------------------
# Stop-word sets used to filter function words from repetition signals.
# Only surface-form tokens are checked (not stems), so the lists must
# include the exact lowercased forms that appear in context.tokens.
# ---------------------------------------------------------------------------

_EN_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "nor", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "as", "into", "through", "about",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "it", "its", "this", "that", "these", "those",
    "i", "we", "you", "he", "she", "they", "me", "him", "her", "us", "them",
    "my", "our", "your", "his", "their",
    "not", "no", "so", "up", "out", "if", "then", "than",
    "there", "here", "when", "where", "who", "which", "what", "how",
    "all", "each", "every", "any", "some", "one", "two", "more", "most",
    "also", "very", "just", "only", "even", "now", "still",
})

_TR_STOP_WORDS: frozenset[str] = frozenset({
    "ve", "bir", "bu", "de", "da", "ki", "ile", "için", "gibi",
    "o", "biz", "siz", "sen", "ben", "onlar",
    "ama", "fakat", "çünkü", "eğer", "ya", "hem",
    "mi", "mı", "mu", "mü", "ne", "her",
    "en", "çok", "az", "daha", "hiç", "bazı",
    "nasıl", "neden", "nerede", "hangi",
    "olan", "olup", "olarak", "ise",
})

_STOP_WORDS: frozenset[str] = _EN_STOP_WORDS | _TR_STOP_WORDS


class RepetitionAnalyzer(BaseAnalyzer[RepetitionResult]):
    """Detects repeated words, phrases, and sentence openers in a document.

    Word repetition is detected via stems so inflectional variants of the
    same root count together (e.g. "technology" and "technologies" are
    one type). Phrase repetition covers bigrams and trigrams from surface
    tokens. Sentence opener repetition tracks the first content word of
    each sentence.

    Stop words are excluded from word and opener detection so that common
    function words ("the", "and", "ve" …) do not contaminate the signal.

    The repetition_score is a weighted combination of three signals mapped
    to [0, 1]:
        0.5 × word signal (excess over-used token ratio)
        0.3 × phrase signal (repeated phrase coverage ratio)
        0.2 × opener signal (repeated opener types / total sentences)
    """

    def __init__(
        self,
        word_min_count: int = 3,
        phrase_min_count: int = 2,
        opening_min_count: int = 2,
    ) -> None:
        """Initialise the analyzer with configurable detection thresholds.

        Args:
            word_min_count: Minimum occurrences for a stem to be flagged.
                Must be >= 2 to satisfy the RepeatedItem.count constraint.
            phrase_min_count: Minimum occurrences for an n-gram to be flagged.
                Must be >= 2.
            opening_min_count: Minimum sentences sharing a starter word for
                it to be flagged. Must be >= 2.

        Raises:
            ValueError: If any threshold is below 2.
        """
        if word_min_count < 2:
            raise ValueError("word_min_count must be >= 2")
        if phrase_min_count < 2:
            raise ValueError("phrase_min_count must be >= 2")
        if opening_min_count < 2:
            raise ValueError("opening_min_count must be >= 2")
        self._word_min = word_min_count
        self._phrase_min = phrase_min_count
        self._opening_min = opening_min_count

    @property
    def name(self) -> str:
        """Unique identifier for this analyzer."""
        return "repetition"

    def analyze(self, context: AnalysisContext) -> RepetitionResult:
        """Detect repetition patterns in the context.

        Args:
            context: Immutable pipeline context from TokenizerService.

        Returns:
            RepetitionResult with detected repeated items and a normalized
            score in [0, 1].
        """
        tokens = context.tokens
        if not tokens:
            return RepetitionResult(
                repeated_words=[],
                repeated_phrases=[],
                repeated_openings=[],
                repetition_score=0.0,
            )

        repeated_words = _find_repeated_words(tokens, context.stems, self._word_min)
        repeated_phrases = _find_repeated_phrases(tokens, self._phrase_min)
        repeated_openings = _find_repeated_openings(
            tokens, context.sentence_token_counts, self._opening_min
        )
        score = _compute_score(
            repeated_words,
            repeated_phrases,
            repeated_openings,
            len(tokens),
            len(context.sentence_token_counts),
        )

        return RepetitionResult(
            repeated_words=repeated_words,
            repeated_phrases=repeated_phrases,
            repeated_openings=repeated_openings,
            repetition_score=score,
        )


# ---------------------------------------------------------------------------
# Module-level pure helper functions — independently testable, no state.
# ---------------------------------------------------------------------------


def _find_repeated_words(
    tokens: tuple[str, ...],
    stems: tuple[str, ...],
    min_count: int,
) -> list[RepeatedItem]:
    """Return content-word stems appearing >= min_count times.

    Groups token positions by their stem. Stop-word surface tokens are
    excluded before grouping. The display text for each item is the most
    frequent surface form that maps to the flagged stem.

    Args:
        tokens: Lowercased surface tokens from the context.
        stems: Morphologically reduced forms parallel to tokens.
        min_count: Minimum occurrence count to flag a stem.

    Returns:
        List of RepeatedItem sorted by count descending.
    """
    stem_positions: dict[str, list[int]] = {}
    for i, (token, stem) in enumerate(zip(tokens, stems, strict=True)):
        if token not in _STOP_WORDS:
            stem_positions.setdefault(stem, []).append(i)

    items: list[RepeatedItem] = []
    for _stem, positions in stem_positions.items():
        if len(positions) >= min_count:
            surface_freq: Counter[str] = Counter(tokens[p] for p in positions)
            display = surface_freq.most_common(1)[0][0]
            items.append(
                RepeatedItem(text=display, count=len(positions), positions=positions)
            )

    return sorted(items, key=lambda x: x.count, reverse=True)


def _find_repeated_phrases(
    tokens: tuple[str, ...],
    min_count: int,
) -> list[RepeatedItem]:
    """Return bigrams and trigrams appearing >= min_count times.

    N-grams composed entirely of stop words are excluded. Results are
    sorted by count descending, then by phrase length descending so that
    longer (more specific) phrases rank above shorter ones of equal count.

    Args:
        tokens: Lowercased surface tokens from the context.
        min_count: Minimum occurrence count to flag a phrase.

    Returns:
        List of RepeatedItem sorted by (count desc, phrase length desc).
    """
    phrase_positions: dict[str, list[int]] = {}
    n = len(tokens)

    for i in range(n - 1):
        bigram = f"{tokens[i]} {tokens[i + 1]}"
        if not _all_stop_words(tokens[i], tokens[i + 1]):
            phrase_positions.setdefault(bigram, []).append(i)

    for i in range(n - 2):
        trigram = f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}"
        if not _all_stop_words(tokens[i], tokens[i + 1], tokens[i + 2]):
            phrase_positions.setdefault(trigram, []).append(i)

    items: list[RepeatedItem] = []
    for phrase, positions in phrase_positions.items():
        if len(positions) >= min_count:
            items.append(
                RepeatedItem(text=phrase, count=len(positions), positions=positions)
            )

    return sorted(
        items,
        key=lambda x: (x.count, len(x.text.split())),
        reverse=True,
    )


def _find_repeated_openings(
    tokens: tuple[str, ...],
    sentence_token_counts: tuple[int, ...],
    min_count: int,
) -> list[str]:
    """Return content words that begin >= min_count sentences.

    Stop words are excluded so that "the", "a", and similar determiners
    do not generate false positives. Results are sorted by frequency
    descending.

    Args:
        tokens: Lowercased surface tokens from the context.
        sentence_token_counts: Per-sentence token counts used to locate
            sentence-starting positions in the flat token tuple.
        min_count: Minimum frequency for a starter word to be flagged.

    Returns:
        List of opener strings sorted by frequency descending.
    """
    opener_freq: Counter[str] = Counter()
    pos = 0
    for count in sentence_token_counts:
        if count > 0 and pos < len(tokens):
            word = tokens[pos]
            if word not in _STOP_WORDS:
                opener_freq[word] += 1
        pos += count

    return [word for word, cnt in opener_freq.most_common() if cnt >= min_count]


def _compute_score(
    repeated_words: list[RepeatedItem],
    repeated_phrases: list[RepeatedItem],
    repeated_openings: list[str],
    total_tokens: int,
    total_sentences: int,
) -> float:
    """Compute a normalized repetition score in [0, 1].

    Combines three weighted signals:
        word signal   (weight 0.5): excess over-repeated tokens / total tokens
        phrase signal (weight 0.3): repeated phrase token coverage / total tokens
        opener signal (weight 0.2): repeated opener types / total sentences

    The raw weighted sum is clamped to [0, 1].

    Args:
        repeated_words: Detected repeated word items.
        repeated_phrases: Detected repeated phrase items.
        repeated_openings: Detected repeated opener strings.
        total_tokens: Total token count in the document.
        total_sentences: Total sentence count in the document.

    Returns:
        Float in [0, 1].
    """
    if total_tokens == 0:
        return 0.0

    excess_tokens = sum(item.count - 1 for item in repeated_words)
    word_signal = excess_tokens / total_tokens

    phrase_token_coverage = sum(
        item.count * len(item.text.split()) for item in repeated_phrases
    )
    phrase_signal = phrase_token_coverage / total_tokens

    opener_signal = len(repeated_openings) / max(1, total_sentences)

    return min(1.0, 0.5 * word_signal + 0.3 * phrase_signal + 0.2 * opener_signal)


def _all_stop_words(*words: str) -> bool:
    """Return True if every word is in the combined stop-word set."""
    return all(w in _STOP_WORDS for w in words)
