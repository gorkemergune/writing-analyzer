"""Configurable weight set for the academic risk scoring model.

Weight rationale
================
Each weight expresses the proportional contribution of that signal to the
overall academic risk score.  Weights must sum to 1.0.  The defaults below
are calibrated for generic academic writing (essays, assignments, reports).

repetition (0.25) — highest weight
    Vocabulary and phrase repetition is the single strongest surface-form
    signal of formulaic writing.  Human writers naturally vary phrasing;
    repetitive text is consistent with template-based or AI-assisted
    writing regardless of domain or language.

low_burstiness (0.20) — second highest
    Uniform sentence length is a well-documented characteristic of
    machine-generated text across languages.  Human writers produce varied
    sentence rhythms; uniformity (low burstiness) is therefore a stronger
    structural signal than any single lexical feature.

transition_overuse (0.15)
    Formulaic discourse markers ("Furthermore", "In conclusion", "Sonuç
    olarak") are template-writing signatures.  They share weight with
    lexical_poverty and cliche_density because they measure overlapping
    dimensions of formulaic language use.

lexical_poverty (0.15)
    Low lexical diversity reduces independently alongside repetition.
    Kept at parity with transition_overuse because restricted vocabulary
    is a strong secondary signal that compounds repetition risk.

cliche_density (0.15)
    Clichés are direct, unambiguous instances of non-original writing.
    Equal weight to transition_overuse and lexical_poverty: each cliché is
    a concrete evidence item, but the signal saturates quickly (five
    clichés per 100 words is already maximal).

readability (0.10) — lowest weight
    Academic text that is unusually easy to read for its register may
    indicate language simplification.  The weakest predictor on its own:
    many legitimate student texts are highly readable, so this feature
    only contributes meaningfully when other signals are also elevated.
"""

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


#: Default weights used when no custom configuration is provided.
DEFAULT_WEIGHTS = ScoringWeights()
