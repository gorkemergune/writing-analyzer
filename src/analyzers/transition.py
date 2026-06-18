"""Transition word and phrase analyzer."""

from collections import Counter

from src.analyzers.base import BaseAnalyzer
from src.models.analysis import AnalysisContext
from src.models.response import TransitionResult

_EN_TRANSITIONS: tuple[str, ...] = (
    "furthermore",
    "moreover",
    "additionally",
    "therefore",
    "consequently",
    "however",
    "nevertheless",
    "in conclusion",
    "in addition",
)

_TR_TRANSITIONS: tuple[str, ...] = (
    "ayrıca",
    "bunun yanında",
    "dolayısıyla",
    "sonuç olarak",
    "özetle",
    "bununla birlikte",
    "ancak",
    "bunun sonucunda",
)

_ALL_TRANSITIONS: tuple[str, ...] = _EN_TRANSITIONS + _TR_TRANSITIONS

_PHRASE_MAP: dict[tuple[str, ...], str] = {
    tuple(phrase.split()): phrase
    for phrase in _ALL_TRANSITIONS
}


class TransitionAnalyzer(BaseAnalyzer[TransitionResult]):
    """Detects transition words and phrases used to signal discourse structure.

    Scans the flat token sequence for every known English and Turkish
    transition expression. High transition density and repeated use of the
    same expressions are signals associated with formulaic writing.

    The transition_score combines two sub-signals mapped to [0, 1]:
        0.6 × density_signal  (transitions per sentence, capped at 2.0)
        0.4 × repeat_ratio    (repeated transitions / unique transitions)
    """

    @property
    def name(self) -> str:
        """Unique identifier for this analyzer."""
        return "transition"

    def analyze(self, context: AnalysisContext) -> TransitionResult:
        """Detect transition expressions in the context.

        Args:
            context: Immutable pipeline context from TokenizerService.

        Returns:
            TransitionResult with counts, unique and repeated transitions,
            density, and a normalized overuse score in [0, 1].
        """
        tokens = context.tokens
        if not tokens:
            return TransitionResult(
                transition_count=0,
                unique_transitions=[],
                repeated_transitions=[],
                transition_density=0.0,
                transition_score=0.0,
            )

        counts = _count_transitions(tokens)
        transition_count = sum(counts.values())
        unique_transitions = sorted(p for p, c in counts.items() if c >= 1)
        repeated_transitions = sorted(p for p, c in counts.items() if c >= 2)

        total_sentences = len(context.sentence_token_counts)
        density = transition_count / max(1, total_sentences)
        score = _compute_score(density, unique_transitions, repeated_transitions)

        return TransitionResult(
            transition_count=transition_count,
            unique_transitions=unique_transitions,
            repeated_transitions=repeated_transitions,
            transition_density=density,
            transition_score=score,
        )



def _count_transitions(tokens: tuple[str, ...]) -> Counter[str]:
    """Return a counter mapping each matched transition phrase to its count.

    Scans every registered phrase against the full token sequence using a
    sliding window. Stop words within multi-word phrases are retained because
    they form part of the transition expression (e.g. "in conclusion").

    Args:
        tokens: Lowercased surface tokens from the context.

    Returns:
        Counter of phrase → occurrence count for all matched transitions.
    """
    counts: Counter[str] = Counter()
    n = len(tokens)
    for phrase_tokens, phrase_str in _PHRASE_MAP.items():
        phrase_len = len(phrase_tokens)
        for i in range(n - phrase_len + 1):
            if tokens[i : i + phrase_len] == phrase_tokens:
                counts[phrase_str] += 1
    return counts


def _compute_score(
    density: float,
    unique_transitions: list[str],
    repeated_transitions: list[str],
) -> float:
    """Compute a normalized transition overuse score in [0, 1].

    Combines two signals:
        density_signal = min(1.0, density / 2.0)
        repeat_ratio   = len(repeated) / len(unique)  — 0 if unique is empty

    Args:
        density: Transition occurrences per sentence.
        unique_transitions: Distinct transition strings detected.
        repeated_transitions: Transitions appearing more than once.

    Returns:
        Float in [0, 1].
    """
    density_signal = min(1.0, density / 2.0)
    repeat_ratio = (
        len(repeated_transitions) / len(unique_transitions)
        if unique_transitions
        else 0.0
    )
    return min(1.0, 0.6 * density_signal + 0.4 * repeat_ratio)
