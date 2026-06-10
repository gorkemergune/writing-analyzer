"""Internal pipeline context model for the Academic Writing Auditor."""

from dataclasses import dataclass

from src.models.enums import DocumentType, Language


@dataclass(frozen=True)
class AnalysisContext:
    """Immutable context object that flows through every analyzer in the pipeline.

    Constructed once by the TokenizerService from an AnalysisRequest, then
    passed to each analyzer as a read-only carrier. frozen=True prevents any
    analyzer from accidentally mutating shared state mid-pipeline.

    Attributes:
        raw_text: Original unmodified input text.
        language: Resolved language (after auto-detection if needed).
        document_type: Document genre from the request.
        cleaned_text: Normalized text after whitespace and encoding cleanup.
        tokens: Lowercased word tokens with punctuation removed.
        sentences: Full sentence strings extracted from the text.
        stems: Morphologically reduced forms of each token in `tokens`.
        sentence_token_counts: Word-token count for each sentence in `sentences`.
    """

    raw_text: str
    language: Language
    document_type: DocumentType
    cleaned_text: str
    tokens: tuple[str, ...]
    sentences: tuple[str, ...]
    stems: tuple[str, ...]
    sentence_token_counts: tuple[int, ...]
