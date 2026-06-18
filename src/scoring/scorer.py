"""Academic risk scorer — aggregates analyzer outputs into a composite score."""

from src.models.enums import RiskLevel
from src.models.response import (
    AcademicRiskScore,
    BurstinessResult,
    ClicheResult,
    ComponentScores,
    ContributionScores,
    ReadabilityResult,
    RepetitionResult,
    SentenceStats,
    TransitionResult,
    WordStats,
)
from src.scoring.weights import DEFAULT_WEIGHTS, ScoringWeights

_READABILITY_BASELINE: float = 40.0
_CONFIDENCE_WORD_THRESHOLD: int = 300
_EXPLANATION_THRESHOLD: float = 40.0

_RISK_THRESHOLDS: tuple[tuple[float, RiskLevel], ...] = (
    (30.0, RiskLevel.LOW),
    (55.0, RiskLevel.MODERATE),
    (75.0, RiskLevel.HIGH),
)


class AcademicRiskScorer:
    """Aggregates individual analyzer outputs into a composite academic risk score.

    The overall score is a weighted sum of six component signals, each
    normalised to [0, 100]:

        overall = Σ weight_i × component_i

    where the weights are supplied via a ScoringWeights instance.

    This class makes no claims about AI authorship.  It reports writing
    quality signals associated with formulaic or repetitive text.

    Usage::

        scorer = AcademicRiskScorer()           # default weights
        result = scorer.score(
            word_stats=..., sentence_stats=...,
            repetition=..., transitions=...,
            burstiness=..., readability=...,
            cliches=...,
        )

    Pass a custom ScoringWeights instance to tune for a specific document
    type or domain.
    """

    def __init__(self, weights: ScoringWeights = DEFAULT_WEIGHTS) -> None:
        """Initialise the scorer with a weight configuration."""
        self._weights = weights

    @property
    def weights(self) -> ScoringWeights:
        """The active weight configuration."""
        return self._weights

    def score(
        self,
        *,
        word_stats: WordStats,
        sentence_stats: SentenceStats,
        repetition: RepetitionResult,
        transitions: TransitionResult,
        burstiness: BurstinessResult,
        readability: ReadabilityResult,
        cliches: ClicheResult,
    ) -> AcademicRiskScore:
        """Compute the composite academic risk score.

        All six analyzer outputs are required.  Passing zero-valued results
        (e.g. for an empty text) is valid and returns a zero-score LOW risk
        result.

        Args:
            word_stats: Output of WordStatisticsAnalyzer.
            sentence_stats: Output of SentenceStatisticsAnalyzer.
            repetition: Output of RepetitionAnalyzer.
            transitions: Output of TransitionAnalyzer.
            burstiness: Output of BurstinessAnalyzer.
            readability: Output of ReadabilityAnalyzer.
            cliches: Output of ClicheAnalyzer.

        Returns:
            AcademicRiskScore with overall_score, risk_level, confidence,
            component_scores, and explanations.
        """
        components = _build_components(
            word_stats=word_stats,
            repetition=repetition,
            transitions=transitions,
            burstiness=burstiness,
            readability=readability,
            cliches=cliches,
        )
        overall = _weighted_sum(components, self._weights)
        contributions = _build_contributions(components, self._weights)
        return AcademicRiskScore(
            overall_score=round(overall, 4),
            risk_level=_classify_risk(overall),
            confidence=_compute_confidence(word_stats.total_words),
            component_scores=components,
            contribution_scores=contributions,
            explanations=_build_explanations(components),
        )



def _build_components(
    *,
    word_stats: WordStats,
    repetition: RepetitionResult,
    transitions: TransitionResult,
    burstiness: BurstinessResult,
    readability: ReadabilityResult,
    cliches: ClicheResult,
) -> ComponentScores:
    """Map each analyzer result to a [0, 100] component score.

    Mappings:
        repetition       = repetition_score × 100
        transition_overuse = transition_score × 100
        low_burstiness   = burstiness_score × 100  (burstiness_score is
                           already inverted: higher = more uniform = more risk)
        lexical_poverty  = (1 − lexical_diversity) × 100
        cliche_density   = cliche_score × 100
        readability      = max(0, readability_score − baseline) /
                           (100 − baseline) × 100
                           (penalises text that is too easy for academic use)

    Args:
        word_stats: Word-level statistics for lexical_poverty.
        repetition: Repetition analysis result.
        transitions: Transition analysis result.
        burstiness: Burstiness analysis result.
        readability: Readability analysis result.
        cliches: Cliché analysis result.

    Returns:
        ComponentScores with all six fields in [0, 100].
    """
    readability_c = (
        max(0.0, readability.readability_score - _READABILITY_BASELINE)
        / (100.0 - _READABILITY_BASELINE)
        * 100.0
    )
    return ComponentScores(
        repetition=repetition.repetition_score * 100.0,
        transition_overuse=transitions.transition_score * 100.0,
        low_burstiness=burstiness.burstiness_score * 100.0,
        lexical_poverty=(1.0 - word_stats.lexical_diversity) * 100.0,
        cliche_density=cliches.cliche_score * 100.0,
        readability=readability_c,
    )


def _weighted_sum(components: ComponentScores, weights: ScoringWeights) -> float:
    """Compute the weighted sum of all component scores.

    Args:
        components: Per-module scores in [0, 100].
        weights: Weight configuration that sums to 1.0.

    Returns:
        Composite score in [0, 100].
    """
    return (
        weights.repetition * components.repetition
        + weights.transition_overuse * components.transition_overuse
        + weights.low_burstiness * components.low_burstiness
        + weights.lexical_poverty * components.lexical_poverty
        + weights.cliche_density * components.cliche_density
        + weights.readability * components.readability
    )


def _build_contributions(
    components: ComponentScores, weights: ScoringWeights
) -> ContributionScores:
    """Compute the weighted contribution of each component to the overall score.

    Each field is component_score × its weight, so all fields sum to
    overall_score.  This lets callers see which module is driving the verdict.

    Args:
        components: Per-module scores in [0, 100].
        weights: Weight configuration that sums to 1.0.

    Returns:
        ContributionScores where each field ∈ [0, 100] and fields sum to
        overall_score.
    """
    return ContributionScores(
        repetition=round(weights.repetition * components.repetition, 4),
        transition_overuse=round(
            weights.transition_overuse * components.transition_overuse, 4
        ),
        low_burstiness=round(weights.low_burstiness * components.low_burstiness, 4),
        lexical_poverty=round(weights.lexical_poverty * components.lexical_poverty, 4),
        cliche_density=round(weights.cliche_density * components.cliche_density, 4),
        readability=round(weights.readability * components.readability, 4),
    )


def _classify_risk(score: float) -> RiskLevel:
    """Map an overall score to a RiskLevel tier.

    Thresholds (from RiskLevel enum):
        ≤ 30  →  LOW
        ≤ 55  →  MODERATE
        ≤ 75  →  HIGH
        > 75  →  VERY_HIGH

    Args:
        score: Overall risk score in [0, 100].

    Returns:
        Corresponding RiskLevel enum value.
    """
    for threshold, level in _RISK_THRESHOLDS:
        if score <= threshold:
            return level
    return RiskLevel.VERY_HIGH


def _compute_confidence(total_words: int) -> float:
    """Estimate assessment confidence from word count.

    Confidence saturates at 1.0 for texts of _CONFIDENCE_WORD_THRESHOLD
    words or more.  Shorter texts receive proportionally lower confidence
    because risk signals are noisier on small samples.

    Args:
        total_words: Total word-token count from WordStats.

    Returns:
        Confidence value in [0.0, 1.0].
    """
    return min(1.0, total_words / _CONFIDENCE_WORD_THRESHOLD)


def _build_explanations(components: ComponentScores) -> list[str]:
    """Generate human-readable explanations for elevated component scores.

    An explanation is added for every component that exceeds
    _EXPLANATION_THRESHOLD (40/100).  Explanations use probabilistic
    language and never assert AI authorship.

    Args:
        components: Per-module scores in [0, 100].

    Returns:
        Sorted list of explanation strings (alphabetical by component name
        for deterministic output).  Empty list when all components are low.
    """
    entries: list[str] = []

    if components.cliche_density > _EXPLANATION_THRESHOLD:
        entries.append(
            f"Clichés and formulaic phrases detected "
            f"({components.cliche_density:.0f}/100): academic clichés "
            f"reduce the originality of the writing."
        )
    if components.lexical_poverty > _EXPLANATION_THRESHOLD:
        entries.append(
            f"Lexical variety is limited "
            f"({components.lexical_poverty:.0f}/100): a restricted "
            f"vocabulary range is associated with formulaic writing."
        )
    if components.low_burstiness > _EXPLANATION_THRESHOLD:
        entries.append(
            f"Sentence rhythm is unusually uniform "
            f"({components.low_burstiness:.0f}/100): human writing "
            f"typically shows greater variation in sentence length."
        )
    if components.readability > _EXPLANATION_THRESHOLD:
        entries.append(
            f"Readability is high for academic writing "
            f"({components.readability:.0f}/100): academic prose "
            f"typically involves greater linguistic complexity."
        )
    if components.repetition > _EXPLANATION_THRESHOLD:
        entries.append(
            f"Repetition signal elevated "
            f"({components.repetition:.0f}/100): vocabulary diversity "
            f"is below the range typical of human academic writing."
        )
    if components.transition_overuse > _EXPLANATION_THRESHOLD:
        entries.append(
            f"Transition word overuse detected "
            f"({components.transition_overuse:.0f}/100): frequent "
            f"formulaic discourse markers reduce textual authenticity."
        )

    return entries
