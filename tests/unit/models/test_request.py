import pytest
from pydantic import ValidationError

from src.models.enums import DocumentType, Language
from src.models.request import AnalysisRequest

_FIFTY_CHARS = "a" * 50
_LONG_TEXT = "This is a sufficiently long text for analysis purposes. " * 4


class TestAnalysisRequestValidInputs:
    def test_all_fields_explicit(self):
        req = AnalysisRequest(
            text=_LONG_TEXT,
            document_type=DocumentType.ACADEMIC,
            language=Language.ENGLISH,
        )
        assert req.text == _LONG_TEXT
        assert req.document_type is DocumentType.ACADEMIC
        assert req.language is Language.ENGLISH

    def test_default_document_type_is_essay(self):
        req = AnalysisRequest(text=_LONG_TEXT)
        assert req.document_type is DocumentType.ESSAY

    def test_default_language_is_none(self):
        req = AnalysisRequest(text=_LONG_TEXT)
        assert req.language is None

    def test_explicit_none_language_accepted(self):
        req = AnalysisRequest(text=_LONG_TEXT, language=None)
        assert req.language is None

    def test_exactly_fifty_chars_is_valid(self):
        req = AnalysisRequest(text=_FIFTY_CHARS)
        assert len(req.text) == 50

    def test_turkish_language_accepted(self):
        req = AnalysisRequest(text=_LONG_TEXT, language=Language.TURKISH)
        assert req.language is Language.TURKISH

    def test_all_document_types_accepted(self):
        for doc_type in DocumentType:
            req = AnalysisRequest(text=_LONG_TEXT, document_type=doc_type)
            assert req.document_type is doc_type

    def test_text_much_longer_than_minimum(self):
        long = "word " * 200
        req = AnalysisRequest(text=long)
        assert len(req.text) > 50

    def test_text_with_unicode_accepted(self):
        unicode_text = "Dijital teknolojinin modern eğitim üzerindeki etkisi büyüktür. " * 3
        req = AnalysisRequest(text=unicode_text, language=Language.TURKISH)
        assert req.language is Language.TURKISH


class TestAnalysisRequestTextValidation:
    def test_forty_nine_chars_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            AnalysisRequest(text="a" * 49)
        assert any(e["loc"] == ("text",) for e in exc_info.value.errors())

    def test_empty_string_raises(self):
        with pytest.raises(ValidationError):
            AnalysisRequest(text="")

    def test_one_char_raises(self):
        with pytest.raises(ValidationError):
            AnalysisRequest(text="x")

    def test_missing_text_raises(self):
        with pytest.raises(ValidationError):
            AnalysisRequest(document_type=DocumentType.ESSAY)  # type: ignore[call-arg]

    def test_text_validation_error_targets_text_field(self):
        with pytest.raises(ValidationError) as exc_info:
            AnalysisRequest(text="short")
        locs = [e["loc"] for e in exc_info.value.errors()]
        assert ("text",) in locs


class TestAnalysisRequestEnumValidation:
    def test_invalid_document_type_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            AnalysisRequest(text=_LONG_TEXT, document_type="thesis")  # type: ignore[arg-type]
        assert any(e["loc"] == ("document_type",) for e in exc_info.value.errors())

    def test_invalid_language_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            AnalysisRequest(text=_LONG_TEXT, language="fr")  # type: ignore[arg-type]
        assert any(e["loc"] == ("language",) for e in exc_info.value.errors())

    def test_integer_language_raises(self):
        with pytest.raises(ValidationError):
            AnalysisRequest(text=_LONG_TEXT, language=42)  # type: ignore[arg-type]

    def test_integer_document_type_raises(self):
        with pytest.raises(ValidationError):
            AnalysisRequest(text=_LONG_TEXT, document_type=1)  # type: ignore[arg-type]


class TestAnalysisRequestSerialization:
    def test_model_dump_contains_all_fields(self):
        req = AnalysisRequest(
            text=_LONG_TEXT,
            document_type=DocumentType.REPORT,
            language=Language.ENGLISH,
        )
        data = req.model_dump()
        assert set(data.keys()) == {"text", "document_type", "language"}

    def test_model_dump_language_none(self):
        req = AnalysisRequest(text=_LONG_TEXT)
        assert req.model_dump()["language"] is None

    def test_json_round_trip_preserves_values(self):
        req = AnalysisRequest(
            text=_LONG_TEXT,
            document_type=DocumentType.ESSAY,
            language=Language.TURKISH,
        )
        restored = AnalysisRequest.model_validate_json(req.model_dump_json())
        assert restored.text == req.text
        assert restored.document_type is req.document_type
        assert restored.language is req.language

    def test_json_round_trip_with_null_language(self):
        req = AnalysisRequest(text=_LONG_TEXT)
        restored = AnalysisRequest.model_validate_json(req.model_dump_json())
        assert restored.language is None

    def test_model_json_schema_has_properties(self):
        schema = AnalysisRequest.model_json_schema()
        assert "properties" in schema
        assert "text" in schema["properties"]
        assert "document_type" in schema["properties"]
        assert "language" in schema["properties"]

    def test_model_json_schema_text_has_min_length(self):
        schema = AnalysisRequest.model_json_schema()
        assert schema["properties"]["text"].get("minLength") == 50

    def test_construct_from_dict(self):
        data = {"text": _LONG_TEXT, "document_type": "academic", "language": "en"}
        req = AnalysisRequest.model_validate(data)
        assert req.document_type is DocumentType.ACADEMIC
        assert req.language is Language.ENGLISH
