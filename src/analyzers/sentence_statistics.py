"""Sentence-level statistics analyzer."""

from src.analyzers.base import BaseAnalyzer
from src.models.analysis import AnalysisContext
from src.models.response import SentenceStats


class SentenceStatisticsAnalyzer(BaseAnalyzer[SentenceStats]):
    """Computes aggregate statistics over the sentence-length distribution.

    Reads only sentence_token_counts from the AnalysisContext —
    pre-computed per-sentence word counts produced by TokenizerService.

    Population variance is used (divides by N, not N-1) because the
    entire document is the complete population, not a sample drawn from
    a larger corpus. This also avoids a ZeroDivisionError for the common
    single-sentence edge case.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this analyzer."""
        return "sentence_stats"

    def analyze(self, context: AnalysisContext) -> SentenceStats:
        """Compute sentence statistics from the context.

        Args:
            context: Immutable pipeline context from TokenizerService.

        Returns:
            SentenceStats with total_sentences, avg_sentence_length,
            sentence_length_variance, min_sentence_length, and
            max_sentence_length. Returns zero-valued result for empty text.
        """
        counts = context.sentence_token_counts
        n = len(counts)

        if n == 0:
            return SentenceStats(
                total_sentences=0,
                avg_sentence_length=0.0,
                sentence_length_variance=0.0,
                min_sentence_length=0,
                max_sentence_length=0,
            )

        mean = sum(counts) / n
        variance = sum((c - mean) ** 2 for c in counts) / n

        return SentenceStats(
            total_sentences=n,
            avg_sentence_length=mean,
            sentence_length_variance=variance,
            min_sentence_length=min(counts),
            max_sentence_length=max(counts),
        )
