"""API response models for the Academic Writing Auditor."""

from pydantic import BaseModel, Field

from src.models.enums import DocumentType, Language, RiskLevel


class RepeatedItem(BaseModel):
    """A word or phrase that appears multiple times in the text.

    Attributes:
        text: The repeated word or phrase.
        count: Number of occurrences. Minimum 2.
        positions: Token indices of each occurrence.
    """

    text: str
    count: int = Field(ge=2, description="Occurrence count. Minimum 2.")
    positions: list[int] = Field(
        description="Token indices where this item occurs.",
    )


class Highlight(BaseModel):
    """A flagged character span within the original text.

    Attributes:
        start: Start character offset (inclusive).
        end: End character offset (exclusive).
        label: Issue category, e.g. 'repeated_phrase' or 'cliche'.
        severity: Severity tier of the flagged issue.
    """

    start: int = Field(ge=0, description="Start character offset (inclusive).")
    end: int = Field(ge=0, description="End character offset (exclusive).")
    label: str = Field(description="Issue category label.")
    severity: RiskLevel


class WordStats(BaseModel):
    """Aggregate statistics computed at the word level.

    Attributes:
        total_words: Total number of word tokens.
        unique_words: Number of distinct word types.
        lexical_diversity: Vocabulary richness score (0–1). Higher is richer.
        avg_word_length: Mean character length of word tokens.
    """

    total_words: int = Field(ge=0)
    unique_words: int = Field(ge=0)
    lexical_diversity: float = Field(
        ge=0.0,
        le=1.0,
        description="Vocabulary richness normalized to 0–1.",
    )
    avg_word_length: float = Field(ge=0.0)


class SentenceStats(BaseModel):
    """Aggregate statistics computed at the sentence level.

    Attributes:
        total_sentences: Total sentence count.
        avg_sentence_length: Mean word count per sentence.
        sentence_length_variance: Variance of per-sentence word counts.
        min_sentence_length: Shortest sentence length in words.
        max_sentence_length: Longest sentence length in words.
    """

    total_sentences: int = Field(ge=0)
    avg_sentence_length: float = Field(ge=0.0)
    sentence_length_variance: float = Field(ge=0.0)
    min_sentence_length: int = Field(ge=0)
    max_sentence_length: int = Field(ge=0)


class RepetitionResult(BaseModel):
    """Output of the repetition analysis module.

    Attributes:
        repeated_words: Individual words appearing above the repetition threshold.
        repeated_phrases: N-gram phrases appearing multiple times.
        repeated_openings: Sentence-initial words or phrases that repeat.
        repetition_score: Normalized repetition signal (0–1). Higher = more repetitive.
    """

    repeated_words: list[RepeatedItem]
    repeated_phrases: list[RepeatedItem]
    repeated_openings: list[str]
    repetition_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized repetition signal. 0 = none, 1 = highly repetitive.",
    )


class TransitionResult(BaseModel):
    """Output of the transition word analysis module.

    Attributes:
        transition_count: Total occurrences of all transition expressions.
        unique_transitions: Distinct transition strings detected, sorted
            alphabetically.
        repeated_transitions: Transitions appearing more than once, sorted
            alphabetically.
        transition_density: Total occurrences divided by sentence count.
        transition_score: Normalized overuse signal in [0, 1]. Higher means
            more formulaic transition use.
    """

    transition_count: int = Field(ge=0)
    unique_transitions: list[str]
    repeated_transitions: list[str]
    transition_density: float = Field(
        ge=0.0,
        description="Transition occurrences per sentence.",
    )
    transition_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized overuse signal. 0 = healthy, 1 = formulaic.",
    )


class BurstinessResult(BaseModel):
    """Output of the burstiness (sentence rhythm) analysis module.

    Burstiness measures sentence-length variability using B = (σ−μ)/(σ+μ).
    Human writing tends to be more varied; formulaic writing is uniform.

    Attributes:
        burstiness_score: Normalized risk score in [0, 1]. Higher means more
            uniform (suspicious). Derived as (1 − B) / 2.
        burstiness_value: Raw B index in [−1, 1]. Negative = uniform,
            positive = bursty.
        classification: Categorical label — one of: very_uniform, uniform,
            neutral, bursty, highly_bursty.
    """

    burstiness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized risk score. 0 = varied writing, 1 = very uniform.",
    )
    burstiness_value: float = Field(
        ge=-1.0,
        le=1.0,
        description="Raw B index: (σ−μ)/(σ+μ). Negative = uniform, positive = bursty.",
    )
    classification: str = Field(
        description="One of: very_uniform, uniform, neutral, bursty, highly_bursty.",
    )


class ReadabilityResult(BaseModel):
    """Output of the readability analysis module.

    English: Flesch Reading Ease (0–100, higher = easier) and Flesch-Kincaid
        Grade Level for the grade label.
    Turkish: Turkish Readability Index (TRI), a simplified heuristic on the
        same 0–100 scale. See analyzers/readability.py for formula and
        documented limitations.

    Attributes:
        readability_score: Score in [0, 100]. Higher = easier to read.
        grade_level: Educational stage label (e.g. "Grade 8" or
            "Lise (High School, Grade 9-12)").
        classification: Difficulty tier — one of: very_difficult, difficult,
            standard, easy, very_easy.
    """

    readability_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Readability score in [0, 100]. Higher = easier.",
    )
    grade_level: str
    classification: str


class ClicheResult(BaseModel):
    """Output of the cliché detection module.

    Attributes:
        detected_cliches: Unique cliché strings found, sorted alphabetically.
        cliche_count: Total occurrences of all detected clichés.
        cliche_density: Total cliché occurrences per 100 word-tokens.
        cliche_score: Normalized overuse signal in [0, 1]. Saturates at a
            density of 5 clichés per 100 words.
    """

    detected_cliches: list[str]
    cliche_count: int = Field(ge=0)
    cliche_density: float = Field(
        ge=0.0,
        description="Cliché occurrences per 100 words.",
    )
    cliche_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized overuse signal. 0 = clean, 1 = highly formulaic.",
    )


class ComponentScores(BaseModel):
    """Individual risk contributions from each analysis module.

    Each field is a 0–100 score where higher means a stronger risk signal.

    Attributes:
        repetition: Risk contribution from the repetition module.
        transition_overuse: Risk contribution from the transition density module.
        low_burstiness: Risk contribution from the burstiness module.
        lexical_poverty: Risk contribution from the lexical diversity module.
        cliche_density: Risk contribution from the cliché detection module.
        readability: Risk contribution from the readability module.
    """

    repetition: float = Field(ge=0.0, le=100.0)
    transition_overuse: float = Field(ge=0.0, le=100.0)
    low_burstiness: float = Field(ge=0.0, le=100.0)
    lexical_poverty: float = Field(ge=0.0, le=100.0)
    cliche_density: float = Field(ge=0.0, le=100.0)
    readability: float = Field(ge=0.0, le=100.0)


class ContributionScores(BaseModel):
    """Weighted contribution of each component to the final score.

    Each field equals component_score × weight, so they sum to overall_score.
    Use this to see which module is driving the risk verdict.

    Attributes:
        repetition: Weighted contribution from the repetition module.
        transition_overuse: Weighted contribution from the transition module.
        low_burstiness: Weighted contribution from the burstiness module.
        lexical_poverty: Weighted contribution from the lexical diversity module.
        cliche_density: Weighted contribution from the cliché detection module.
        readability: Weighted contribution from the readability module.
    """

    repetition: float = Field(ge=0.0, le=100.0)
    transition_overuse: float = Field(ge=0.0, le=100.0)
    low_burstiness: float = Field(ge=0.0, le=100.0)
    lexical_poverty: float = Field(ge=0.0, le=100.0)
    cliche_density: float = Field(ge=0.0, le=100.0)
    readability: float = Field(ge=0.0, le=100.0)


class AcademicRiskScore(BaseModel):
    """Aggregated academic risk assessment.

    This score reflects writing quality signals associated with formulaic or
    repetitive writing. It does not assert AI authorship with certainty.

    Attributes:
        overall_score: Weighted composite score (0–100). Higher = more risk signals.
        risk_level: Categorical risk tier derived from overall_score.
        confidence: Assessment confidence (0–1). Lower for short texts.
        component_scores: Per-module score breakdown.
        contribution_scores: Weighted contribution of each module (sums to
            overall_score).
        explanations: Human-readable descriptions of elevated risk signals.
            Empty when no component exceeds the explanation threshold.
    """

    overall_score: float = Field(ge=0.0, le=100.0)
    risk_level: RiskLevel
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Assessment confidence. Lower for short texts.",
    )
    component_scores: ComponentScores
    contribution_scores: ContributionScores
    explanations: list[str]


class AnalysisReport(BaseModel):
    """Complete analysis report returned to the API client.

    Attributes:
        language: Detected or specified language.
        document_type: Document genre used for scoring context.
        word_stats: Word-level statistics.
        sentence_stats: Sentence-level statistics.
        repetition: Repetition analysis results.
        transitions: Transition word analysis results.
        burstiness: Sentence rhythm analysis results.
        readability: Readability analysis results.
        cliches: Cliché detection results.
        academic_risk: Composite risk score and breakdown.
        highlights: Character-level spans flagging issues in the original text.
        suggestions: Actionable improvement recommendations.
        processing_time_ms: Wall-clock time taken to produce this report.
    """

    language: Language
    document_type: DocumentType
    word_stats: WordStats
    sentence_stats: SentenceStats
    repetition: RepetitionResult
    transitions: TransitionResult
    burstiness: BurstinessResult
    readability: ReadabilityResult
    cliches: ClicheResult
    academic_risk: AcademicRiskScore
    highlights: list[Highlight]
    suggestions: list[str]
    processing_time_ms: float = Field(ge=0.0)
