"""Burstiness analyzer for sentence-rhythm and length variability."""

import statistics

from src.analyzers.base import BaseAnalyzer
from src.models.analysis import AnalysisContext
from src.models.response import BurstinessResult

_THRESHOLDS: tuple[float, float, float, float] = (-0.6, -0.2, 0.2, 0.5)
_LABELS: tuple[str, str, str, str, str] = (
    "very_uniform",
    "uniform",
    "neutral",
    "bursty",
    "highly_bursty",
)


class BurstinessAnalyzer(BaseAnalyzer[BurstinessResult]):
    """Measures sentence-length variability using the burstiness index B.

    Formula: B = (σ − μ) / (σ + μ)
        σ = population standard deviation of per-sentence word counts
        μ = mean per-sentence word count
        B ∈ [−1, 1]: negative → uniform, zero → neutral, positive → bursty

    burstiness_score is the normalized risk contribution for the scoring
    pipeline: score = (1 − B) / 2, mapping B = −1 → 1.0 (high risk, very
    uniform) and B = 1 → 0.0 (low risk, highly varied).

    Texts with fewer than two sentences receive a neutral default because
    intra-document rhythm cannot be measured from a single sentence.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this analyzer."""
        return "burstiness"

    def analyze(self, context: AnalysisContext) -> BurstinessResult:
        """Compute the burstiness index and risk score for the context.

        Args:
            context: Immutable pipeline context from TokenizerService.

        Returns:
            BurstinessResult with burstiness_value, burstiness_score, and
            a categorical classification label.
        """
        lengths = context.sentence_token_counts
        if len(lengths) < 2:
            return BurstinessResult(
                burstiness_score=0.5,
                burstiness_value=0.0,
                classification="neutral",
            )

        b_value = _compute_burstiness(lengths)
        score = (1.0 - b_value) / 2.0
        return BurstinessResult(
            burstiness_score=score,
            burstiness_value=b_value,
            classification=_classify(b_value),
        )



def _compute_burstiness(lengths: tuple[int, ...]) -> float:
    """Compute B = (σ − μ) / (σ + μ) using population standard deviation.

    Returns 0.0 for the degenerate case where μ + σ = 0 (all zero-length
    sentences), which maps to the neutral midpoint.

    Args:
        lengths: Per-sentence word-token counts (at least two elements).

    Returns:
        Float in [−1, 1].
    """
    mu = statistics.mean(lengths)
    sigma = statistics.pstdev(lengths)
    denom = sigma + mu
    if denom == 0.0:
        return 0.0
    return (sigma - mu) / denom


def _classify(b_value: float) -> str:
    """Map a burstiness index to one of five categorical labels.

    Thresholds (strict upper bounds):
        B < -0.6  → very_uniform
        B < -0.2  → uniform
        B <  0.2  → neutral
        B <  0.5  → bursty
        otherwise → highly_bursty

    Args:
        b_value: Raw B index in [−1, 1].

    Returns:
        One of: very_uniform, uniform, neutral, bursty, highly_bursty.
    """
    if b_value < -0.6:
        return "very_uniform"
    if b_value < -0.2:
        return "uniform"
    if b_value < 0.2:
        return "neutral"
    if b_value < 0.5:
        return "bursty"
    return "highly_bursty"
