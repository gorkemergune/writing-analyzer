"""Word-level statistics analyzer."""

from src.analyzers.base import BaseAnalyzer
from src.models.analysis import AnalysisContext
from src.models.response import WordStats


class WordStatisticsAnalyzer(BaseAnalyzer[WordStats]):
    """Computes aggregate word-level statistics from an AnalysisContext.

    Unique vocabulary is counted over stemmed/lemmatized forms so that
    inflectional variants of the same root (e.g. "runs", "running") are
    treated as a single type. This gives a more accurate measure of lexical
    breadth than counting surface forms. Average word length is computed
    from surface-form tokens because the reader experiences the full form.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this analyzer."""
        return "word_stats"

    def analyze(self, context: AnalysisContext) -> WordStats:
        """Compute word statistics from the context.

        Args:
            context: Immutable pipeline context from TokenizerService.

        Returns:
            WordStats with total_words, unique_words, lexical_diversity,
            and avg_word_length. Returns zero-valued result for empty text.
        """
        tokens = context.tokens
        stems = context.stems
        total = len(tokens)

        if total == 0:
            return WordStats(
                total_words=0,
                unique_words=0,
                lexical_diversity=0.0,
                avg_word_length=0.0,
            )

        unique = len(set(stems))
        diversity = unique / total
        avg_length = sum(len(t) for t in tokens) / total

        return WordStats(
            total_words=total,
            unique_words=unique,
            lexical_diversity=diversity,
            avg_word_length=avg_length,
        )
