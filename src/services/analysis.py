"""AnalysisService — orchestrates the full text analysis pipeline."""

import re
import time

from src.analyzers.burstiness import BurstinessAnalyzer
from src.analyzers.cliche import ClicheAnalyzer
from src.analyzers.readability import ReadabilityAnalyzer
from src.analyzers.repetition import RepetitionAnalyzer
from src.analyzers.sentence_statistics import SentenceStatisticsAnalyzer
from src.analyzers.transition import TransitionAnalyzer
from src.analyzers.word_statistics import WordStatisticsAnalyzer
from src.models.enums import RiskLevel
from src.models.request import AnalysisRequest
from src.models.response import (
    AcademicRiskScore,
    AnalysisReport,
    ClicheResult,
    Highlight,
    RepetitionResult,
    TransitionResult,
)
from src.scoring.scorer import AcademicRiskScorer
from src.services.language_detector import LanguageDetector
from src.services.tokenizer import TokenizerService

_SUGGESTION_THRESHOLD: float = 40.0

_SUGGESTIONS: dict[str, str] = {
    "cliche_density": (
        "Replace formulaic phrases such as 'in conclusion' and 'it is important "
        "to note that' with precise language that reflects your specific argument."
    ),
    "lexical_poverty": (
        "Expand vocabulary range: consult discipline-specific glossaries and avoid "
        "using the same descriptors in close proximity."
    ),
    "low_burstiness": (
        "Vary sentence lengths to create a more natural prose rhythm. "
        "Mix short, direct statements with longer analytical sentences."
    ),
    "readability": (
        "Academic writing benefits from greater syntactic complexity. "
        "Consider more subordinate clauses and discipline-appropriate terminology."
    ),
    "repetition": (
        "Reduce word repetition by using synonyms, pronouns, or restructured "
        "sentences when referring back to concepts already introduced."
    ),
    "transition_overuse": (
        "Reduce the frequency of transition phrases. Let logical sentence structure "
        "carry the argument without formulaic signposting."
    ),
}

_PRI_CLICHE: int = 0
_PRI_TRANSITION: int = 1
_PRI_PHRASE: int = 2

_MAX_PHRASE_HIGHLIGHTS: int = 5


class AnalysisService:
    """Orchestrates the full academic writing analysis pipeline.

    Accepts an AnalysisRequest, resolves the language, builds an
    AnalysisContext, runs all seven analyzers in sequence, scores the
    results, and returns a complete AnalysisReport.

    All dependencies are injected through the constructor so the class
    can be used in any context (web server, CLI, test) without change.
    Passing ``None`` for any parameter causes the production default to
    be constructed on first use.

    Example::

        service = AnalysisService()
        report  = service.analyze(
            AnalysisRequest(text="...", document_type="essay")
        )
    """

    def __init__(
        self,
        *,
        language_detector: LanguageDetector | None = None,
        tokenizer: TokenizerService | None = None,
        word_stats_analyzer: WordStatisticsAnalyzer | None = None,
        sentence_stats_analyzer: SentenceStatisticsAnalyzer | None = None,
        repetition_analyzer: RepetitionAnalyzer | None = None,
        transition_analyzer: TransitionAnalyzer | None = None,
        burstiness_analyzer: BurstinessAnalyzer | None = None,
        readability_analyzer: ReadabilityAnalyzer | None = None,
        cliche_analyzer: ClicheAnalyzer | None = None,
        scorer: AcademicRiskScorer | None = None,
    ) -> None:
        """Initialise with optional dependency overrides.

        Args:
            language_detector: Detects language when not specified in request.
            tokenizer: Builds AnalysisContext from raw text.
            word_stats_analyzer: Computes word-level statistics.
            sentence_stats_analyzer: Computes sentence-level statistics.
            repetition_analyzer: Detects repeated words and phrases.
            transition_analyzer: Detects transition word overuse.
            burstiness_analyzer: Measures sentence-length variability.
            readability_analyzer: Estimates text readability.
            cliche_analyzer: Detects formulaic cliché phrases.
            scorer: Aggregates analyzer outputs into a composite risk score.
        """
        self._detector = language_detector or LanguageDetector()
        self._tokenizer = tokenizer or TokenizerService()
        self._word_stats = word_stats_analyzer or WordStatisticsAnalyzer()
        self._sentence_stats = sentence_stats_analyzer or SentenceStatisticsAnalyzer()
        self._repetition = repetition_analyzer or RepetitionAnalyzer()
        self._transition = transition_analyzer or TransitionAnalyzer()
        self._burstiness = burstiness_analyzer or BurstinessAnalyzer()
        self._readability = readability_analyzer or ReadabilityAnalyzer()
        self._cliche = cliche_analyzer or ClicheAnalyzer()
        self._scorer = scorer or AcademicRiskScorer()

    def analyze(self, request: AnalysisRequest) -> AnalysisReport:
        """Run the full analysis pipeline on the request text.

        Steps:
            1. Resolve language: use ``request.language`` or auto-detect.
            2. Build AnalysisContext via TokenizerService.
            3. Execute all seven analyzers in order.
            4. Score results with AcademicRiskScorer.
            5. Build character-level highlights and actionable suggestions.
            6. Return an AnalysisReport with wall-clock timing.

        Args:
            request: Validated analysis request.

        Returns:
            AnalysisReport containing all analyzer outputs, the composite
            risk score, character-level highlights, suggestions, and
            processing time in milliseconds.
        """
        start_ns = time.perf_counter_ns()

        language = request.language or self._detector.detect(request.text)
        context = self._tokenizer.build_context(
            request.text, language, request.document_type
        )

        word_stats = self._word_stats.analyze(context)
        sentence_stats = self._sentence_stats.analyze(context)
        repetition = self._repetition.analyze(context)
        transitions = self._transition.analyze(context)
        burstiness = self._burstiness.analyze(context)
        readability = self._readability.analyze(context)
        cliches = self._cliche.analyze(context)

        academic_risk = self._scorer.score(
            word_stats=word_stats,
            sentence_stats=sentence_stats,
            repetition=repetition,
            transitions=transitions,
            burstiness=burstiness,
            readability=readability,
            cliches=cliches,
        )

        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000

        return AnalysisReport(
            language=language,
            document_type=request.document_type,
            word_stats=word_stats,
            sentence_stats=sentence_stats,
            repetition=repetition,
            transitions=transitions,
            burstiness=burstiness,
            readability=readability,
            cliches=cliches,
            academic_risk=academic_risk,
            highlights=_build_highlights(
                context.cleaned_text,
                repetition,
                transitions,
                cliches,
                academic_risk,
            ),
            suggestions=_build_suggestions(academic_risk),
            processing_time_ms=elapsed_ms,
        )


def _find_spans(text: str, phrase: str) -> list[tuple[int, int]]:
    """Return (start, end) spans for all non-overlapping matches in text.

    Uses case-insensitive matching so "in conclusion" finds "In Conclusion"
    in the source text.  ``re.finditer`` guarantees non-overlapping results.

    Args:
        text: Source text to search (original form, not lowercased).
        phrase: Phrase to locate; matched case-insensitively.

    Returns:
        List of ``(start, end)`` character offset tuples in document order.
    """
    return [
        (m.start(), m.end())
        for m in re.finditer(re.escape(phrase), text, re.IGNORECASE)
    ]


def _score_to_risk(component_score: float) -> RiskLevel:
    """Map a [0, 100] component score to a RiskLevel for highlight severity.

    Uses the same boundaries as AcademicRiskScorer._classify_risk so that
    severity labels are consistent across the report.

    Args:
        component_score: Per-module score in [0, 100].

    Returns:
        Corresponding RiskLevel.
    """
    if component_score <= 30.0:
        return RiskLevel.LOW
    if component_score <= 55.0:
        return RiskLevel.MODERATE
    if component_score <= 75.0:
        return RiskLevel.HIGH
    return RiskLevel.VERY_HIGH


def _build_highlights(
    text: str,
    repetition: RepetitionResult,
    transitions: TransitionResult,
    cliches: ClicheResult,
    academic_risk: AcademicRiskScore,
) -> list[Highlight]:
    """Build character-level highlight spans from analyzer outputs.

    Detects three categories of spans in the cleaned text:

        ``cliche``           — every occurrence of a detected cliché phrase.
        ``transition_overuse`` — every occurrence of a transition used >1×.
        ``repeated_phrase``  — occurrences of the top-5 repeated n-grams.

    When two spans overlap, the higher-priority label wins:
    ``cliche > transition_overuse > repeated_phrase``.  Spans are returned
    in document order (ascending start position).

    Args:
        text: Cleaned source text; character offsets reference this string.
        repetition: Output of RepetitionAnalyzer.
        transitions: Output of TransitionAnalyzer.
        cliches: Output of ClicheAnalyzer.
        academic_risk: Composite score used to derive per-label severity.

    Returns:
        Non-overlapping Highlight list in document order.
    """
    entries: list[tuple[int, int, str, RiskLevel, int]] = []

    cliche_sev = _score_to_risk(academic_risk.component_scores.cliche_density)
    for phrase in cliches.detected_cliches:
        for start, end in _find_spans(text, phrase):
            entries.append((start, end, "cliche", cliche_sev, _PRI_CLICHE))

    tr_sev = _score_to_risk(academic_risk.component_scores.transition_overuse)
    for phrase in transitions.repeated_transitions:
        for start, end in _find_spans(text, phrase):
            entries.append((start, end, "transition_overuse", tr_sev, _PRI_TRANSITION))

    rep_sev = _score_to_risk(academic_risk.component_scores.repetition)
    for item in repetition.repeated_phrases[:_MAX_PHRASE_HIGHLIGHTS]:
        for start, end in _find_spans(text, item.text):
            entries.append((start, end, "repeated_phrase", rep_sev, _PRI_PHRASE))

    entries.sort(key=lambda e: (e[0], e[4]))

    result: list[Highlight] = []
    covered_end = -1
    for start, end, label, severity, _ in entries:
        if start >= covered_end:
            result.append(
                Highlight(start=start, end=end, label=label, severity=severity)
            )
            covered_end = end

    return result


def _build_suggestions(academic_risk: AcademicRiskScore) -> list[str]:
    """Generate actionable writing suggestions for elevated risk signals.

    A suggestion is added for each component that exceeds
    ``_SUGGESTION_THRESHOLD`` (40/100).  The order mirrors
    ``_build_explanations`` in scorer.py so that explanations and
    suggestions remain aligned in the report.

    Args:
        academic_risk: Composite score with per-module component breakdown.

    Returns:
        List of actionable suggestion strings.  Empty when all components
        are at or below the threshold.
    """
    cs = academic_risk.component_scores
    suggestions: list[str] = []

    if cs.cliche_density > _SUGGESTION_THRESHOLD:
        suggestions.append(_SUGGESTIONS["cliche_density"])
    if cs.lexical_poverty > _SUGGESTION_THRESHOLD:
        suggestions.append(_SUGGESTIONS["lexical_poverty"])
    if cs.low_burstiness > _SUGGESTION_THRESHOLD:
        suggestions.append(_SUGGESTIONS["low_burstiness"])
    if cs.readability > _SUGGESTION_THRESHOLD:
        suggestions.append(_SUGGESTIONS["readability"])
    if cs.repetition > _SUGGESTION_THRESHOLD:
        suggestions.append(_SUGGESTIONS["repetition"])
    if cs.transition_overuse > _SUGGESTION_THRESHOLD:
        suggestions.append(_SUGGESTIONS["transition_overuse"])

    return suggestions
