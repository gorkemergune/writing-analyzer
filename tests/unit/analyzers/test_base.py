import pytest

from src.analyzers.base import BaseAnalyzer
from src.models.analysis import AnalysisContext
from src.models.enums import DocumentType, Language


# ---------------------------------------------------------------------------
# Minimal concrete implementations used only within this test module
# ---------------------------------------------------------------------------

class _StringAnalyzer(BaseAnalyzer[str]):
    """Concrete analyzer that returns the language name as a string."""

    @property
    def name(self) -> str:
        return "string_analyzer"

    def analyze(self, context: AnalysisContext) -> str:
        return context.language.value


class _IntAnalyzer(BaseAnalyzer[int]):
    """Concrete analyzer that returns the token count as an int."""

    @property
    def name(self) -> str:
        return "int_analyzer"

    def analyze(self, context: AnalysisContext) -> int:
        return len(context.tokens)


def _make_context(language: Language = Language.ENGLISH) -> AnalysisContext:
    return AnalysisContext(
        raw_text="Test text.",
        language=language,
        document_type=DocumentType.ESSAY,
        cleaned_text="Test text.",
        tokens=("test", "text"),
        sentences=("Test text.",),
        stems=("test", "text"),
        sentence_token_counts=(2,),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBaseAnalyzerAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseAnalyzer()  # type: ignore[abstract]

    def test_subclass_without_name_raises(self):
        with pytest.raises(TypeError):

            class _MissingName(BaseAnalyzer[str]):
                def analyze(self, context: AnalysisContext) -> str:
                    return ""

            _MissingName()

    def test_subclass_without_analyze_raises(self):
        with pytest.raises(TypeError):

            class _MissingAnalyze(BaseAnalyzer[str]):
                @property
                def name(self) -> str:
                    return "missing_analyze"

            _MissingAnalyze()

    def test_subclass_must_implement_both_methods(self):
        with pytest.raises(TypeError):

            class _Empty(BaseAnalyzer[str]):
                pass

            _Empty()


class TestBaseAnalyzerConcrete:
    def test_string_analyzer_name(self):
        analyzer = _StringAnalyzer()
        assert analyzer.name == "string_analyzer"

    def test_int_analyzer_name(self):
        analyzer = _IntAnalyzer()
        assert analyzer.name == "int_analyzer"

    def test_string_analyzer_returns_language_code(self):
        ctx = _make_context(language=Language.ENGLISH)
        result = _StringAnalyzer().analyze(ctx)
        assert result == "en"

    def test_string_analyzer_turkish(self):
        ctx = _make_context(language=Language.TURKISH)
        result = _StringAnalyzer().analyze(ctx)
        assert result == "tr"

    def test_int_analyzer_returns_token_count(self):
        ctx = _make_context()
        result = _IntAnalyzer().analyze(ctx)
        assert result == 2

    def test_int_analyzer_empty_tokens(self):
        ctx = AnalysisContext(
            raw_text="",
            language=Language.ENGLISH,
            document_type=DocumentType.ESSAY,
            cleaned_text="",
            tokens=(),
            sentences=(),
            stems=(),
            sentence_token_counts=(),
        )
        result = _IntAnalyzer().analyze(ctx)
        assert result == 0

    def test_analyze_does_not_mutate_context(self):
        ctx = _make_context()
        original_tokens = ctx.tokens
        _StringAnalyzer().analyze(ctx)
        assert ctx.tokens == original_tokens

    def test_analyze_called_multiple_times_is_consistent(self):
        ctx = _make_context()
        analyzer = _StringAnalyzer()
        assert analyzer.analyze(ctx) == analyzer.analyze(ctx)


class TestBaseAnalyzerGenericType:
    def test_string_result_type(self):
        result = _StringAnalyzer().analyze(_make_context())
        assert isinstance(result, str)

    def test_int_result_type(self):
        result = _IntAnalyzer().analyze(_make_context())
        assert isinstance(result, int)

    def test_different_analyzers_independent(self):
        ctx = _make_context()
        str_result = _StringAnalyzer().analyze(ctx)
        int_result = _IntAnalyzer().analyze(ctx)
        assert isinstance(str_result, str)
        assert isinstance(int_result, int)
