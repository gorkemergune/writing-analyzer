import pytest

from src.models.enums import Language
from src.services.language_detector import LanguageDetectionError, LanguageDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detector_with(code: str) -> LanguageDetector:
    """Return a LanguageDetector whose internal function always returns `code`."""
    return LanguageDetector(_detect_fn=lambda _: code)


def _detector_raising(error: Exception) -> LanguageDetector:
    """Return a LanguageDetector whose internal function always raises `error`."""

    def _raise(_: str) -> str:
        raise error

    return LanguageDetector(_detect_fn=_raise)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestLanguageDetectorInstantiation:
    def test_default_instantiation(self):
        detector = LanguageDetector()
        assert detector is not None

    def test_custom_fallback(self):
        detector = LanguageDetector(fallback=Language.TURKISH)
        assert detector._fallback is Language.TURKISH

    def test_default_fallback_is_english(self):
        detector = LanguageDetector()
        assert detector._fallback is Language.ENGLISH

    def test_inject_detect_fn(self):
        detector = _detector_with("en")
        assert detector is not None


# ---------------------------------------------------------------------------
# Supported language detection
# ---------------------------------------------------------------------------

class TestLanguageDetectorSupportedCodes:
    def test_detect_english_code(self):
        assert _detector_with("en").detect("text") is Language.ENGLISH

    def test_detect_turkish_code(self):
        assert _detector_with("tr").detect("metin") is Language.TURKISH

    def test_detect_returns_language_enum(self):
        result = _detector_with("en").detect("any text")
        assert isinstance(result, Language)

    def test_detect_english_with_long_text(self):
        long_text = "word " * 100
        assert _detector_with("en").detect(long_text) is Language.ENGLISH


# ---------------------------------------------------------------------------
# Unsupported codes — fallback behaviour
# ---------------------------------------------------------------------------

class TestLanguageDetectorFallback:
    def test_unsupported_code_returns_default_fallback(self):
        result = _detector_with("de").detect("Guten Tag")
        assert result is Language.ENGLISH

    def test_unsupported_code_returns_custom_fallback(self):
        detector = LanguageDetector(
            fallback=Language.TURKISH,
            _detect_fn=lambda _: "fr",
        )
        assert detector.detect("Bonjour") is Language.TURKISH

    def test_unknown_two_letter_code_returns_fallback(self):
        assert _detector_with("xx").detect("text") is Language.ENGLISH

    def test_empty_string_code_returns_fallback(self):
        assert _detector_with("").detect("text") is Language.ENGLISH

    def test_numeric_code_returns_fallback(self):
        assert _detector_with("42").detect("text") is Language.ENGLISH

    def test_mixed_case_code_returns_fallback(self):
        # Language codes are lowercase; "EN" is not a valid value
        assert _detector_with("EN").detect("text") is Language.ENGLISH


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------

class TestLanguageDetectorErrors:
    def test_detection_error_propagates(self):
        detector = _detector_raising(LanguageDetectionError("failed"))
        with pytest.raises(LanguageDetectionError):
            detector.detect("text")

    def test_unexpected_exception_propagates(self):
        detector = _detector_raising(RuntimeError("unexpected"))
        with pytest.raises(RuntimeError):
            detector.detect("text")

    def test_detection_error_message_preserved(self):
        detector = _detector_raising(LanguageDetectionError("bad input"))
        with pytest.raises(LanguageDetectionError, match="bad input"):
            detector.detect("text")


# ---------------------------------------------------------------------------
# Supported codes constant
# ---------------------------------------------------------------------------

class TestSupportedCodes:
    def test_all_language_values_are_supported(self):
        from src.services.language_detector import _SUPPORTED_CODES

        for lang in Language:
            assert lang.value in _SUPPORTED_CODES

    def test_supported_codes_count_matches_language_enum(self):
        from src.services.language_detector import _SUPPORTED_CODES

        assert len(_SUPPORTED_CODES) == len(Language)
