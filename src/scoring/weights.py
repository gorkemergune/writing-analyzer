"""Scoring weight configuration for the academic risk model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringWeights:
    """Immutable weight configuration for AcademicRiskScorer.

    All six weights must sum to 1.0 (enforced to three decimal places by
    __post_init__).  Pass a custom instance to AcademicRiskScorer to tune
    the model for a specific document type or domain.

    Attributes:
        repetition: Weight for the repetition signal (default 0.25).
        transition_overuse: Weight for transition density (default 0.15).
        low_burstiness: Weight for sentence-length uniformity (default 0.20).
        lexical_poverty: Weight for low lexical diversity (default 0.15).
        cliche_density: Weight for cliché presence (default 0.15).
        readability: Weight for excess readability (default 0.10).
    """

    repetition: float = 0.25
    transition_overuse: float = 0.15
    low_burstiness: float = 0.20
    lexical_poverty: float = 0.15
    cliche_density: float = 0.15
    readability: float = 0.10

    def __post_init__(self) -> None:
        """Validate that weights sum to 1.0."""
        total = (
            self.repetition
            + self.transition_overuse
            + self.low_burstiness
            + self.lexical_poverty
            + self.cliche_density
            + self.readability
        )
        if not (0.999 <= total <= 1.001):
            raise ValueError(
                f"ScoringWeights must sum to 1.0 (got {total:.6f}).  "
                "Adjust the weights so their sum equals 1.0."
            )


DEFAULT_WEIGHTS = ScoringWeights()
