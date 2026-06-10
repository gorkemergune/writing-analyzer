"""Language detection service for the Academic Writing Auditor."""

from collections.abc import Callable

from src.models.enums import Language

_SUPPORTED_CODES: frozenset[str] = frozenset(lang.value for lang in Language)


class LanguageDetectionError(Exception):
    """Raised when language detection fails irrecoverably (e.g. empty text)."""


class LanguageDetector:
    """Detects the language of a text and maps it to a supported Language.

    An optional detection callable is accepted so the class can be tested
    without the langdetect dependency. In production, the default
    ``langdetect.detect`` is used.

    Attributes:
        _fallback: Language returned when the detected code is not supported.
        _detect_fn: Callable that maps a text string to a BCP-47 language code.
    """

    def __init__(
        self,
        fallback: Language = Language.ENGLISH,
        _detect_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._fallback = fallback
        self._detect_fn = _detect_fn if _detect_fn is not None else self._langdetect

    @staticmethod
    def _langdetect(text: str) -> str:
        """Invoke langdetect and re-raise library errors as LanguageDetectionError."""
        from langdetect import detect  # type: ignore[import-untyped]
        from langdetect.lang_detect_exception import LangDetectException  # type: ignore[import-untyped]

        try:
            return detect(text)
        except LangDetectException as exc:
            raise LanguageDetectionError(f"Detection failed: {exc}") from exc

    def detect(self, text: str) -> Language:
        """Detect the language of the text.

        Args:
            text: Input text to detect. Should be at least a few sentences for
                reliable results with the default langdetect backend.

        Returns:
            The detected Language, or the configured fallback when the detected
            code is not in the supported set.

        Raises:
            LanguageDetectionError: If the underlying detection call fails.
        """
        code = self._detect_fn(text)
        if code in _SUPPORTED_CODES:
            return Language(code)
        return self._fallback
